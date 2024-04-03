# TODO

- Disable buttons while training is ongoing (this should also fix the 4 tensor leaks)
- Show what's being fed when training, ideally animate it
- find a way in the UI to switch between models.
- Error in console when resizing

# Maybe

- Simplify the link filtering thing, work started on another "betterer branch"
- Highlight connections when hovering over a neuron.
- Essayer de suivre le réseau à l’envers pour voir les pixels qui représentent un digit
- Entrainer avec un batch de 1 et montrer le gradient.
- Montrer que le système ne peut pas expliquer comment il prend les décisions. Changer les poids et voir ce qui se passe.
- Séparer accuracy sur entrainement VS jeu de test. Montrer que ça se trompe, éventuellement en choisissant un jeu de test très différent
- Idée de ne pas montrer un certain chiffre pendant l’entrainement, ou moins le montrer
- Idée d'entrainer avec un symbole supplémentaire que le réseau reconnait sans qu'on le sache vraiment.
