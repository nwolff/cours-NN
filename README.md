A web application to experiment with a neural network that recognizes handwritten digits : training, predicting, evaluating the precision.

Deployed automatically when branch main is pushed:

http://rn.nwolff.info/

The python backend is only needed if you want to train new/different models

# Tooling

Requires an installation of nodejs

## Installing dependencies

    npm install

## Developing

Running the app under development, with automatic reload :

    npm run dev -- --open

Automatically formatting :

    npm run format

Type-checking :

    npm run check

There are currently some typing errors, many because we extract numbers from tensors that are very generically typed.

## Verifying the production build

Sveltekit has a server-side rendering capability to optimize page loads.
We want to build a single page app that will be served statically, so we have to disable server-side rendering for each of our routes.
This step makes sure we haven't forgotten anything.

    npm run build
    npm run preview -- --open

## Deploying to GitHub pages manually

(The app is currently automatically deployed with github actions when the main branch is pushed)

This will run a full build and deploy

    npm run gh-pages

# Refs

https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0
