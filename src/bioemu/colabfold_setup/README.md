# Automated colabfold embeds.

1. (Optional). Set `COLABFOLD_DIR` environment variable to where you have a [`localcolabfold`](https://github.com/YoshitakaMo/localcolabfold) installation, or where you'd like it installed. If not set, this defaults to `~/.localcolabfold`. If manually specified, you may want to set it in your `~/.bashrc` permanently.
2. Run `setup.sh`.
3. Run sampling code. `get_embeds.get_colabfold_embeds` returns paths to single and pair evoformer embeddings.