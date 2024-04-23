## TODO

- Rename formulaire to questionnaire (on front page)
- Zero and One : Only enable training buttons when there is a drawing
- Show what's being fed when training : Either as an overlay on the training zone, or as pixels on the network
- Disable buttons while training is ongoing or queue them up. Maybe show current progress
  (this should also fix 4 tensor leaks)
- Think about removing the distribution-chart and making the prediction clearer on the network itself
- Show biases when no activations (so we can talk about biases)

## Technical

- Bug on firefox when training the zalando model, the accuracy stays very low and the confusion matrix shows a single line, as if there was only one number shown. Two participants had this on firefox and the problem went away when they switched to chrome
- Something with https over mobile networks
- Extract common things from all routes pages: functions, components, slots ?
- Error in console when resizing
- Tighter typing
- Try to remove the extra invert-canvas, by using a grayscale and invert filter on the normalize canvas
  (had this idea _after_ rewriting some image processing in javascript)
- Simplify link filtering
- Use tf.confusionMatrix, instead of doing it manually
