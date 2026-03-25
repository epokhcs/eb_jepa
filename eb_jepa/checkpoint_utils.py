"""Utilities for loading models from checkpoints."""

import torch
import torch.nn as nn
import yaml
from pathlib import Path

from eb_jepa.architectures import (
    ImpalaEncoder,
    RNNPredictor,
    DiscreteActionEncoder,
    Projector,
    DiscreteInverseDynamicsModel,
    InverseDynamicsModel,
    RewardPredictionHead,
)
from eb_jepa.datasets.registry import EnvironmentRegistry
from eb_jepa.datasets.utils import load_env_data_config
from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import VC_IDM_Sim_Regularizer

logger = get_logger(__name__)


def load_jepa_from_checkpoint(checkpoint_path: str, config_path: str, device='auto'):
    """
    Load a trained JEPA model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint (.pth.tar)
        config_path: Path to training config (.yaml)
        device: Device to load model on ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        jepa: Loaded JEPA model
        cfg: Configuration namespace
        data_config: Data configuration
    """
    # Load config
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Convert to namespace
    from argparse import Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    cfg = dict_to_namespace(cfg_dict)

    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load data config
    merged_cfg = load_env_data_config(cfg.data.env_name, vars(cfg.data))
    config_class = EnvironmentRegistry.get_config_class(cfg.data.env_name)
    data_config = update_config_from_yaml(config_class, merged_cfg)

    # Get action space info
    dummy_env = EnvironmentRegistry.create_env(cfg.data.env_name, data_config)
    action_space_info = dummy_env.get_action_space_info()
    dummy_env.close() if hasattr(dummy_env, 'close') else None

    is_discrete = action_space_info['type'] == 'discrete'
    action_dim = action_space_info.get('n', action_space_info.get('dim', 2))

    # Build encoder
    logger.info("Building encoder...")
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )

    # Build action encoder
    if is_discrete:
        embedding_dim = getattr(cfg.model, 'action_embedding_dim', 64)
        aencoder = DiscreteActionEncoder(num_actions=action_dim, embedding_dim=embedding_dim)
        predictor_action_dim = embedding_dim
        logger.info(f"Using discrete action encoder: {action_dim} actions -> {embedding_dim}D")
    else:
        aencoder = nn.Identity()
        predictor_action_dim = action_dim
        logger.info(f"Using continuous actions: {action_dim}D")

    # Build predictor
    logger.info("Building predictor...")
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        final_ln=encoder.final_ln,
        action_encoder=aencoder,
        action_dim=predictor_action_dim,
    )

    # Build projector (if used)
    if cfg.model.regularizer.use_proj:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None

    # Build IDM
    logger.info("Building IDM...")
    if is_discrete:
        idm_model = DiscreteInverseDynamicsModel(
            state_dim=encoder.mlp_output_dim,
            hidden_dim=256,
            num_actions=action_dim,
        )
    else:
        idm_model = InverseDynamicsModel(
            feat_dim=encoder.mlp_output_dim,
            action_dim=action_dim,
        )

    # Build regularizer
    logger.info("Building regularizer...")
    # Remove keys not accepted by VC_IDM_Sim_Regularizer
    reg_args = vars(cfg.model.regularizer).copy()
    reg_args.pop('use_proj', None)
    regularizer = VC_IDM_Sim_Regularizer(
        idm=idm_model,
        projector=projector,
        **reg_args
    )

    # Build reward head if enabled
    reward_head = None
    if getattr(cfg.model, 'reward_prediction', False):
        logger.info("Building reward head...")
        reward_head = RewardPredictionHead(
            state_dim=encoder.mlp_output_dim,
            hidden_dim=getattr(cfg.model, 'reward_head_hidden_dim', 256),
        )

    # Build JEPA model
    logger.info("Building JEPA model...")
    jepa = JEPA(
        encoder=encoder,
        aencoder=aencoder,
        predictor=predictor,
        regularizer=regularizer,
        predcost=None,  # or set appropriately if needed
    )

    # Attach reward head as attribute if enabled
    if reward_head is not None:
        jepa.reward_head = reward_head

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state, stripping _orig_mod. prefix if present
    state_dict = checkpoint['model_state_dict']
    # Remove _orig_mod. prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    # Map aencoder.* keys to action_encoder.* for compatibility
    mapped_state_dict = {}
    for k, v in state_dict.items():
        # Map aencoder.* to action_encoder.*
        if k.startswith('aencoder.'):
            mapped_state_dict['action_encoder.' + k[len('aencoder.'):]] = v
        # Map predictor.action_encoder.embedding.weight to action_encoder.embedding.weight
        elif k == 'predictor.action_encoder.embedding.weight':
            mapped_state_dict['action_encoder.embedding.weight'] = v
        # Map _orig_mod.predictor.action_encoder.embedding.weight to action_encoder.embedding.weight
        elif k == '_orig_mod.predictor.action_encoder.embedding.weight':
            mapped_state_dict['action_encoder.embedding.weight'] = v
        else:
            mapped_state_dict[k] = v
    # If predictor.action_encoder.embedding.weight is missing but action_encoder.embedding.weight exists, copy it
    if (
        'predictor.action_encoder.embedding.weight' not in mapped_state_dict
        and 'action_encoder.embedding.weight' in mapped_state_dict
    ):
        mapped_state_dict['predictor.action_encoder.embedding.weight'] = mapped_state_dict['action_encoder.embedding.weight']
    jepa.load_state_dict(mapped_state_dict)
    jepa.to(device)
    jepa.eval()

    logger.info(f"✅ Model loaded successfully from epoch {checkpoint['epoch']}, step {checkpoint['step']}")

    return jepa, cfg, data_config
