# TODO

- Carry statistics and history of models and display them (number of samples seen and accuracy)
- Check for tensor leaks https://www.tensorflow.org/js/guide/tensors_operations
- Diagramme performance en Français et avec des graphiques plus clairs
- Training and operating views show a confusing mix of things.
  They should most likely be completely separate.
  Also simplify the link filtering thing, work started on another "betterer branch"
- Show what's being fed when training, ideally animate it.

# Maybe

- Highlight connections when hovering over a neuron.
- Essayer de suivre le réseau à l’envers pour voir les pixels qui représentent un digit
- Entrainer avec un batch de 1 et montrer le gradient.
- Montrer que le système ne peut pas expliquer comment il prend les décisions. Changer les poids et voir ce qui se passe. Technique pour voir la sensibilité aux entrées. Montrer qu’on réagit à des lignes.
- Séparer accuracy sur entrainement VS jeu de test. Montrer que ça se trompe, éventuellement en choisissant un jeu de test très différents
- Idée de ne pas montrer un certain chiffre pendant l’entrainement, ou moins le montrer
