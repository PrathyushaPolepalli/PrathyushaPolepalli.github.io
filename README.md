# Systems for AI

Source for [prathyushapolepalli.github.io](https://prathyushapolepalli.github.io), a long-form research notebook about GPU systems, distributed AI, LLM inference, and production infrastructure.

## Design and dependency policy

The site is intentionally built with GitHub Pages' native Jekyll support, custom local CSS, and custom local JavaScript. It has no third-party theme, Jekyll plugin, hook, package dependency, or remote browser script.

## Write a post

1. Copy `POST_TEMPLATE.md` to `_posts/YYYY-MM-DD-short-title.md`.
2. Replace the front matter and remove sections that do not apply.
3. Put original diagrams and benchmark images under `assets/images/<post-slug>/`.
4. Link primary sources and record versions, hardware, workload, and measurement methodology.
5. If an approved Jekyll installation is already available, preview with `jekyll serve`.

GitHub Pages builds the site after changes reach the publishing branch; no local dependency installation is required.

## Publish the account site

The repository must be public and named exactly:

```text
PrathyushaPolepalli/PrathyushaPolepalli.github.io
```

Push the `main` branch, then configure **Settings > Pages > Build and deployment** to deploy from the `main` branch at `/ (root)`. The site URL will be:

```text
https://prathyushapolepalli.github.io
```
