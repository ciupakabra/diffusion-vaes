@jax.vmap
def kl(mu, logvar):
  return -0.5 * jnp.sum(1. + logvar - mu**2. - jnp.exp(logvar))

class VanillaVAE(hk.Module):
    """ Vanilla VAE model """

    def __init__(self, 
            latent_size: int,
            encoder: hk.Module,
            decoder: hk.Module,
            likelihood: str = BERNOULLI,
            ):
        super().__init__()

        self.likelihood = likelihood
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, y : jnp.array, is_training : bool) -> jnp.array:
        enc = self.encoder(y, is_training)
        mu, logvar = jnp.split(enc, 2, axis=1)
        return mu, logvar

    def decode(self, x : jnp.array, is_training : bool) -> jnp.array:
        return self.decoder(x, is_training)

    def sample(self, n: int, is_training : bool) -> jnp.array:
        x = jax.random.normal(hk.next_rng_key(), (n, self.latent_size))
        logits = self.decode(x, is_training)
        return jax.nn.sigmoid(logits)

    def elbo(self, y : jnp.array, is_training : bool) -> jnp.array:
        mu, logvar = self.encode(y, is_training)
        x = mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), mu.shape)
        logits = self.decode(x, is_training)

        kl_loss = kl(mu, logvar)
        recon_loss = log_likelihood(logits, y, distribution=self.likelihood)

        return jnp.mean(recon_loss - kl_loss)
