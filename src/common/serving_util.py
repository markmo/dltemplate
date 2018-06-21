from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model
import os
import tensorflow as tf


class TfModelDeployer(object):

    def __init__(self, app_name, predict_op_name='predict_op:0', input_name='input_X:0', slo_micros=3000000):
        """

        :param app_name: (str) The unique name of the application.
        :param predict_op_name:
        :param input_name:
        :param slo_micros: (int) The service level objective for the application in microseconds.
        """
        self._app_name = app_name
        self._predict_op_name = predict_op_name
        self._input_name = input_name
        self._slo_micros = slo_micros
        self.conn = create_clipper_conn()

    def deploy_model_from_checkpoint(self, model_name, model_dir, version='1',
                                     input_type='floats', default_output='-1.0',
                                     meta_file=None, predict_fn=None):
        if meta_file is None:
            meta_file = '{}.meta'.format(model_name)

        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        self.deploy_model(sess, model_name, version, input_type, default_output, predict_fn)

    def deploy_model(self, sess, model_name, version='1', input_type='floats',
                     default_output='-1.0', predict_fn=None):
        """

        :param model_name:
        :param predict_fn:
        :param sess:
        :param version:
        :param input_type: (str) The type of the request data this
               endpoint can process. Input type can be one of “integers”,
               “floats”, “doubles”, “bytes”, or “strings”.
        :param default_output: (str) The default output for the application.
        :return:
        """
        if predict_fn is None:
            predict_op_name = self._predict_op_name
            input_name = self._input_name

            def predict(sess_, inputs):
                print('received input of shape:', inputs.shape, 'dtype:', inputs.dtype)
                print('predict_op_name:', predict_op_name, type(predict_op_name),
                      'input_name:', input_name, type(input_name))
                return sess_.run(predict_op_name, feed_dict={input_name: inputs})

            predict_fn = predict

        self.conn.register_application(self._app_name, input_type, default_output, self._slo_micros)
        deploy_tensorflow_model(self.conn, model_name, version, input_type, predict_fn, sess)
        self.conn.link_model_to_app(app_name=self._app_name, model_name=model_name)
        return self.conn.get_query_addr()

    def get_query_addr(self):
        return self.conn.get_query_addr()

    def shutdown(self):
        self.conn.stop_all_model_containers()
        self.conn.delete_application(self._app_name)
        self.conn.stop_all()


def create_clipper_conn():
    conn = ClipperConnection(DockerContainerManager())
    conn.start_clipper()
    return conn


def freeze_graph(model_dir, output_node_names, output_name):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError("Export directory %s doesn't exist." % model_dir)

    if not output_node_names:
        print("Please supply the name of a node to --output_node_names")
        return -1

    # retrieve the checkpoint path
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # specify path of the frozen graph
    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = '{}/{}.pb'.format(absolute_model_dir, output_name)

    # clear devices for TensorFlow to control which device to load operations
    clear_devices = True

    # start session with a temporary fresh graph
    with tf.Session(graph=tf.Graph()) as sess:
        # import meta graph into the current default graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # restore weights
        saver.restore(sess, input_checkpoint)

        # use a built-in helper to export variables as constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # session used to retrieve weights
            tf.get_default_graph().as_graph_def(),  # the graph definition used to retrieve the nodes
            output_node_names.split(',')  # filter selected nodes on the given output node names
        )

        # finally serialize and dump the output graph
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        print('%d ops in the final graph.' % len(output_graph_def.node))

    return output_graph_def


def load_graph(frozen_graph_filename, prefix=''):
    # load the protobuf file from disk and parse to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graph_def into a new graph and return it
    with tf.Graph().as_default() as graph:
        # the name var will prefix every op/node in your graph
        tf.import_graph_def(graph_def, name=prefix)

    return graph


def serve(app_name, model_name, model_dir, version,
          predict_op_name, input_name, input_type,
          default_output, meta_file=None):
    deployer = TfModelDeployer(app_name, predict_op_name, input_name)
    # noinspection PyBroadException
    try:
        deployer.deploy_model_from_checkpoint(model_name, model_dir, version, input_type, default_output, meta_file)
    except Exception as e:
        print('Err deploying app: ', app_name, e)

    input('Press Enter to shutdown...')
    deployer.shutdown()
