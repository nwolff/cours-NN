# TODO

- Disable buttons while training is ongoing
- Show what's being fed when training, ideally animate it
- Work with 0 and 1 models, find a UI thingy (maybe a url parameter ?) to select which one

# Maybe

- Simplify the link filtering thing, work started on another "betterer branch"
- Highlight connections when hovering over a neuron.
- Essayer de suivre le réseau à l’envers pour voir les pixels qui représentent un digit
- Entrainer avec un batch de 1 et montrer le gradient.
- Montrer que le système ne peut pas expliquer comment il prend les décisions. Changer les poids et voir ce qui se passe.
- Séparer accuracy sur entrainement VS jeu de test. Montrer que ça se trompe, éventuellement en choisissant un jeu de test très différent
- Idée de ne pas montrer un certain chiffre pendant l’entrainement, ou moins le montrer
- Idée d'entrainer avec un symbole supplémentaire que le réseau reconnait sans qu'on le sache vraiment.

# Technical

- Memory leaks on training: One on every run + 4 when there is already a training running
- Error in console when resizing
