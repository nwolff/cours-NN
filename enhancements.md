## TODO

- Ajouter le nombre d'éléments qu'on a vus pour le test
- Show what's being fed when training : Either as an overlay on the training zone, or as pixels on the network
- Disable buttons while automatic training is ongoing or queue them up. Maybe show current progress
  (this should also fix 4 tensor leaks)

## Technical

- Store the all digits dataset locally
- Simplify link filtering
- Bug on firefox when training the Zalando model, the accuracy stays very low and the confusion matrix shows a single line, as if there was only one number shown. Two participants had this on firefox and the problem went away when they switched to chrome
- Tighter typing
- Use tf.confusionMatrix, instead of doing it manually
- Extract common things from all routes pages: functions, components, slots ?
- Error in console when resizing
- Try to remove the extra invert-canvas, by using a grayscale and invert filter on the normalize canvas
  (had this idea _after_ rewriting some image processing in javascript)
