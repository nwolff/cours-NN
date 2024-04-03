# Setting up dependencies

## Install project dependencies

    allow direnv

    pip install -U pip
    pip install -Ur dev-requirements.txt

# Preparing datasets

We use the full MNIST dataset, but also want a trimmed-down version with only zeroes and ones.

Because it's painful to do the trimming in the frontend (masking data is asynchronous and that mixes really badly with the manual memory management required by tensors),
we prepare that dataset here in python and then use it from the frontend.

    ./make_zero_one_dataset.py

    cp build/zero_one* ../static

# Training

The idea is to show pre-trained models in the frontend, in order to validate/compare with the
model currently being trained.

## Training the models

The first model is very simple and trains quickly, the second one is much slower

    ./tune_and_train_all_digits.py
    ./train_lenet.py

## Converting trained models for the frontend to use

     tensorflowjs_converter --input_format=keras build/all_digits.keras \
     ../static/all_digits

     tensorflowjs_converter --input_format=keras build/le_net.keras \
     ../static/le_net
