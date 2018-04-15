from common.util import next_batch, plot_accuracy
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def train(data, constants, input_placeholders, optimizer, loss_op, model, y_, minibatch=True):
    batch_size = constants['batch_size']
    n_epochs = constants['n_epochs']
    n_report_steps = constants['n_report_steps']
    x_train = data['X_train']
    y_train = data['y_train']
    x_val = data['X_val']
    y_val = data['y_val']
    x_test = data['X_test']
    y_test = data['y_test']
    m = x_train.shape[0]
    input_x, input_y = input_placeholders
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_costs = []
        val_costs = []
        for epoch in range(n_epochs):
            avg_cost = 0.
            if minibatch:
                n_batches = int(m / batch_size)
                for i in range(n_batches):
                    x_batch, y_batch = next_batch(x_train, y_train, batch_size)
                    _, cost = sess.run([optimizer, loss_op],
                                       feed_dict={input_x: x_batch, input_y: y_batch})
                    avg_cost = cost / n_batches

                train_costs.append(sess.run(loss_op, feed_dict={input_x: x_train, input_y: y_train}))
            else:
                _, cost = sess.run([optimizer, loss_op],
                                   feed_dict={input_x: x_train, input_y: y_train})
                avg_cost = cost
                train_costs.append(cost)

            val_costs.append(sess.run(loss_op, feed_dict={input_x: x_val, input_y: y_val}))

            if epoch % n_report_steps == 0:
                print('Epoch:', '%04d,' % (epoch + 1), 'cost={:.9f}'.format(avg_cost))

            # simple impl of early stopping
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            val_acc = accuracy.eval({input_x: x_val, input_y: y_val})
            if val_acc > constants['early_stop_threshold']:
                break

        plot_accuracy(n_epochs, train_costs, val_costs)

        correct_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))

        print('Train accuracy:', accuracy.eval({input_x: x_train, input_y: y_train}))
        print('Test accuracy:', accuracy.eval({input_x: x_test, input_y: y_test}))
        print('Test AUC:', roc_auc_score(y_test, sess.run(y_, {input_x: x_test})))


def train_using_estimator(data, model_builder, constants):
    model = tf.estimator.Estimator(model_builder)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X_train']}, y=data['y_train'],
        batch_size=constants['batch_size'], num_epochs=None, shuffle=True
        )

    model.train(input_fn, steps=constants['n_epochs'])

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X_test']}, y=data['y_test'],
        batch_size=constants['batch_size'], shuffle=False
        )

    metrics = model.evaluate(input_fn)

    return metrics
