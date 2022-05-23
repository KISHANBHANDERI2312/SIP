import kivy.app
import classif

class FirstApp(kivy.app.App):
    def classify_image(self):
        img_path = self.root.ids["img"].source
        # img_features = Fruits.extract_features(img_path)
        predicted_class = classif.predictop(img_path)
        self.root.ids["label"].text = "Predicted Class : " + predicted_class
firstApp = FirstApp(title="Cifar10 Recognition.")
firstApp.run()