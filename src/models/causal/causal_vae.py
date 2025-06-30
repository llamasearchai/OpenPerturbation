"""Causal Variational Autoencoder for perturbation biology."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Tuple, Optional, List
import numpy as np


class CausalEncoder(nn.Module):
    """Encoder that separates causal and confounding factors."""

    def __init__(
        self, input_dim: int, latent_dim: int, causal_dim: int, hidden_dims: List[int] = [512, 256]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.causal_dim = causal_dim
        self.confounding_dim = latent_dim - causal_dim

        # Shared encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim
        self.shared_encoder = nn.Sequential(*layers)

        # Causal factor encoder
        self.causal_mu = nn.Linear(prev_dim, causal_dim)
        self.causal_logvar = nn.Linear(prev_dim, causal_dim)

        # Confounding factor encoder
        self.confounding_mu = nn.Linear(prev_dim, self.confounding_dim)
        self.confounding_logvar = nn.Linear(prev_dim, self.confounding_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input into causal and confounding factors."""
        # Shared encoding
        h = self.shared_encoder(x)

        # Causal factors (interventionable)
        z_causal_mu = self.causal_mu(h)
        z_causal_logvar = self.causal_logvar(h)

        # Confounding factors (observational)
        z_conf_mu = self.confounding_mu(h)
        z_conf_logvar = self.confounding_logvar(h)

        return {
            "z_causal_mu": z_causal_mu,
            "z_causal_logvar": z_causal_logvar,
            "z_confounding_mu": z_conf_mu,
            "z_confounding_logvar": z_conf_logvar,
        }


class CausalDecoder(nn.Module):
    """Decoder that reconstructs from causal and confounding factors."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent factors to reconstruction."""
        return self.decoder(z)


class InterventionModule(nn.Module):
    """Module for modeling interventions on causal factors."""

    def __init__(self, causal_dim: int, intervention_types: int = 3):
        super().__init__()
        self.causal_dim = causal_dim
        self.intervention_types = intervention_types

        # Intervention effect predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(causal_dim + intervention_types, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, causal_dim),
        )

        # Intervention strength modulator
        self.strength_modulator = nn.Sequential(
            nn.Linear(intervention_types, 32), nn.ReLU(), nn.Linear(32, causal_dim), nn.Sigmoid()
        )

    def forward(
        self,
        z_causal: torch.Tensor,
        intervention_type: torch.Tensor,
        intervention_strength: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply intervention to causal factors."""
        # Concatenate causal factors with intervention type
        intervention_input = torch.cat([z_causal, intervention_type], dim=1)

        # Predict intervention effect
        intervention_effect = self.intervention_predictor(intervention_input)

        # Modulate by intervention strength
        if intervention_strength is None:
            intervention_strength = self.strength_modulator(intervention_type)

        # Apply intervention
        z_causal_intervened = z_causal + intervention_strength * intervention_effect

        return z_causal_intervened


class CausalVAE(nn.Module):
    """Causal Variational Autoencoder for perturbation biology."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Model dimensions
        self.input_dim = config["input_dim"]
        self.latent_dim = config["latent_dim"]
        self.causal_dim = config["causal_dim"]
        self.confounding_dim = self.latent_dim - self.causal_dim

        # Model components
        self.encoder = CausalEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            causal_dim=self.causal_dim,
            hidden_dims=config.get("encoder_hidden_dims", [512, 256]),
        )

        self.decoder = CausalDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.input_dim,
            hidden_dims=config.get("decoder_hidden_dims", [256, 512]),
        )

        self.intervention_module = InterventionModule(
            causal_dim=self.causal_dim, intervention_types=config.get("intervention_types", 3)
        )

        # Loss weights
        self.beta_kl = config.get("beta_kl", 1.0)
        self.gamma_causal = config.get("gamma_causal", 1.0)
        self.lambda_intervention = config.get("lambda_intervention", 1.0)

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input into causal and confounding factors."""
        encoding = self.encoder(x)

        # Sample from distributions
        z_causal = self._reparameterize(encoding["z_causal_mu"], encoding["z_causal_logvar"])
        z_confounding = self._reparameterize(
            encoding["z_confounding_mu"], encoding["z_confounding_logvar"]
        )

        return {
            "z_causal": z_causal,
            "z_confounding": z_confounding,
            "z_causal_mu": encoding["z_causal_mu"],
            "z_causal_logvar": encoding["z_causal_logvar"],
            "z_confounding_mu": encoding["z_confounding_mu"],
            "z_confounding_logvar": encoding["z_confounding_logvar"],
        }

    def decode(self, z_causal: torch.Tensor, z_confounding: torch.Tensor) -> torch.Tensor:
        """Decode causal and confounding factors to reconstruction."""
        z = torch.cat([z_causal, z_confounding], dim=1)
        return self.decoder(z)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self, x: torch.Tensor, intervention_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CausalVAE."""
        # Encode
        encoding = self.encode(x)
        z_causal = encoding["z_causal"]
        z_confounding = encoding["z_confounding"]

        # Decode (factual)
        x_recon = self.decode(z_causal, z_confounding)

        outputs = {
            "x_recon": x_recon,
            "z_causal": z_causal,
            "z_confounding": z_confounding,
            "z_causal_mu": encoding["z_causal_mu"],
            "z_causal_logvar": encoding["z_causal_logvar"],
            "z_confounding_mu": encoding["z_confounding_mu"],
            "z_confounding_logvar": encoding["z_confounding_logvar"],
        }

        # Apply intervention if specified (counterfactual)
        if intervention_type is not None:
            z_causal_intervened = self.intervention_module(z_causal, intervention_type)
            x_counterfactual = self.decode(z_causal_intervened, z_confounding)

            outputs.update(
                {"z_causal_intervened": z_causal_intervened, "x_counterfactual": x_counterfactual}
            )

        return outputs

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        intervention_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute CausalVAE loss components."""
        batch_size = x.size(0)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs["x_recon"], x, reduction="sum") / batch_size

        # KL divergence losses
        causal_prior = Normal(0, 1)
        causal_posterior = Normal(
            outputs["z_causal_mu"], torch.exp(0.5 * outputs["z_causal_logvar"])
        )
        kl_causal = kl_divergence(causal_posterior, causal_prior).sum(dim=1).mean()

        conf_prior = Normal(0, 1)
        conf_posterior = Normal(
            outputs["z_confounding_mu"], torch.exp(0.5 * outputs["z_confounding_logvar"])
        )
        kl_confounding = kl_divergence(conf_posterior, conf_prior).sum(dim=1).mean()

        kl_loss = kl_causal + kl_confounding

        # Causal disentanglement loss
        causal_disentangle_loss = self._compute_disentanglement_loss(
            outputs["z_causal"], outputs["z_confounding"]
        )

        # Intervention consistency loss
        intervention_loss = torch.tensor(0.0, device=x.device)
        if "x_counterfactual" in outputs and intervention_labels is not None:
            intervention_effect = torch.norm(
                outputs["x_counterfactual"] - outputs["x_recon"], dim=1
            )
            intervention_strength = torch.norm(intervention_labels, dim=1)
            intervention_loss = F.mse_loss(intervention_effect, intervention_strength)

        # Total loss
        total_loss = (
            recon_loss
            + self.beta_kl * kl_loss
            + self.gamma_causal * causal_disentangle_loss
            + self.lambda_intervention * intervention_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_causal": kl_causal,
            "kl_confounding": kl_confounding,
            "disentanglement_loss": causal_disentangle_loss,
            "intervention_loss": intervention_loss,
        }

    def _compute_disentanglement_loss(
        self, z_causal: torch.Tensor, z_confounding: torch.Tensor
    ) -> torch.Tensor:
        """Compute disentanglement loss to encourage factor independence."""
        z_causal_centered = z_causal - z_causal.mean(dim=0, keepdim=True)
        z_conf_centered = z_confounding - z_confounding.mean(dim=0, keepdim=True)

        cross_corr = torch.mm(z_causal_centered.t(), z_conf_centered) / z_causal.size(0)
        disentangle_loss = torch.norm(cross_corr, p="fro") ** 2

        return disentangle_loss

    def generate_counterfactuals(
        self, x: torch.Tensor, intervention_types: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Generate counterfactual examples for different interventions."""
        self.eval()
        counterfactuals = []

        with torch.no_grad():
            encoding = self.encode(x)
            z_causal = encoding["z_causal"]
            z_confounding = encoding["z_confounding"]

            for intervention_type in intervention_types:
                z_causal_intervened = self.intervention_module(z_causal, intervention_type)
                x_counterfactual = self.decode(z_causal_intervened, z_confounding)
                counterfactuals.append(x_counterfactual)

        return counterfactuals

    def get_causal_factors(self, x: torch.Tensor) -> torch.Tensor:
        """Extract causal factors for downstream analysis."""
        self.eval()
        with torch.no_grad():
            encoding = self.encode(x)
            return encoding["z_causal"]

    def predict_intervention_effect(
        self, x: torch.Tensor, intervention_type: torch.Tensor
    ) -> torch.Tensor:
        """Predict the effect of a specific intervention."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, intervention_type)

            effect = outputs["x_counterfactual"] - outputs["x_recon"]
            effect_magnitude = torch.norm(effect, dim=1)

            return effect_magnitude


class CausalDiscoveryModule(nn.Module):
    """Module for discovering causal relationships in perturbation data."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Causal VAE for representation learning
        self.causal_vae = CausalVAE(config["causal_vae"])

        # Graph neural network for causal structure
        causal_dim = config["causal_vae"]["causal_dim"]
        self.causal_graph = nn.Parameter(torch.zeros(causal_dim, causal_dim))

        # Causal mechanism predictors
        self.mechanism_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(causal_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(causal_dim)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Discover causal structure and mechanisms."""
        # Extract causal factors
        causal_factors = self.causal_vae.get_causal_factors(x)

        # Predict causal graph structure
        causal_adjacency = torch.sigmoid(self.causal_graph)

        # Predict causal mechanisms
        mechanisms = []
        for i, mechanism_net in enumerate(self.mechanism_networks):
            # Input: all other causal factors (parents)
            parents = causal_factors * causal_adjacency[i].unsqueeze(0)
            mechanism_output = mechanism_net(parents)
            mechanisms.append(mechanism_output)

        mechanisms = torch.cat(mechanisms, dim=1)

        return {
            "causal_factors": causal_factors,
            "causal_graph": causal_adjacency,
            "mechanisms": mechanisms,
        }
