from common.util_keras import ModelSaveCallback, TqdmProgressCallback


last_finished_epoch = None


def train(model, data, constants):
    x_train = data['X_train']
    x_test = data['X_test']
    model.fit(x=x_train, y=x_train, epochs=constants['n_epochs'],
              validation_data=[x_test, x_test],
              callbacks=[ModelSaveCallback(constants['model_filename']), TqdmProgressCallback()],
              verbose=0,
              initial_epoch=last_finished_epoch or 0)
