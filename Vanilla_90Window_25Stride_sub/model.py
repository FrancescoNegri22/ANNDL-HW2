import os
import tensorflow as tf
import numpy as np

TELESCOPE = 9
WINDOW_SIZE = 90
model_names = ["Vanilla_A", "Vanilla_B", "Vanilla_C",
               "Vanilla_D", "Vanilla_E", "Vanilla_F",]


class model:
    def __init__(self, path):
        self.models = {}
        for c in model_names:
            self.models[c] = tf.keras.models.load_model(
                os.path.join(path, "SubmissionModel", c))

    # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
    def predict(self, X, categories):
        # Note: this is just an example.
        # Here the model.predict is called
        # out = self.model.predict(X)

        out = np.array([])
        batch_size = X.shape[0]

        for i in range(batch_size):
            batch = X[i, -WINDOW_SIZE:]

            model = self.models["Vanilla_" + categories[i]]
            X_temp = batch.reshape((1, WINDOW_SIZE, 1))
            predictions = []
            for _ in range(0, TELESCOPE):
                current_prediction = model.predict(X_temp)
                pred = current_prediction[0, 0, 0]

                if len(predictions) == 0:
                    predictions = np.array([pred])
                else:
                    predictions = np.append(predictions, pred)

                X_temp = np.append(X_temp[0][1:], pred).reshape(
                    (1, WINDOW_SIZE, 1))

            predictions = predictions.reshape((1, 9))
            if (out.shape[0] == 0):
                out = predictions
            else:
                out = np.vstack((out, predictions))

        out = tf.convert_to_tensor(out, dtype=tf.float32)

        return out
