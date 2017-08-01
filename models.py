import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)

def _cnn_to_dist(convs, hiddens, num_atoms, dueling, inpt, num_actions, scope, reuse = False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        logits = []
        dists = []
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("dist_value"):
            dist_out = out
            for hidden in hiddens:
                dist_out = layers.fully_connected(dist_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            dist_out = layers.fully_connected(dist_out, num_outputs=num_actions*num_atoms, activation_fn=None)


            for action in range(num_actions):
                logits = dist_out[:, num_atoms * action : num_atoms * (action + 1)]
                print "start : {}, end : {}".format(num_atoms * action, num_atoms * (action + 1))
                prob = tf.nn.softmax(logits)
                dists.append(prob)

        # print np.shape(dist[1])

        dist = tf.concat(dists, axis=1)
        print np.shape(dist)

        return dist


        # if dueling:
        #     with tf.variable_scope("state_value"):
        #         state_out = out
        #         for hidden in hiddens:
        #             state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
        #         state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
        #     action_scores_mean = tf.reduce_mean(action_scores, 1)
        #     action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
        #     return state_score + action_scores_centered
        # else:
        #     return action_scores
        # return out
 
def cnn_to_dist(convs, hiddens, num_atoms, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_dist(convs, hiddens, num_atoms, dueling, *args, **kwargs)




def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores
        return out


def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)

