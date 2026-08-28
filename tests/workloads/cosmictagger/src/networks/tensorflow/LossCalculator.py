import tensorflow as tf

class LossCalculator(object):

    def __init__(self, balance_type=None, channels_dim=1):

        object.__init__(self)


        if balance_type not in ["focal", "light", "even", "none"] and balance_type is not None:
            raise Exception("Unsupported loss balancing recieved: ", balance_type)

        self.balance_type = balance_type
        self.channels_dim = channels_dim

        if balance_type != "none":
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
        else:
            self._criterion = tf.nn.sparse_softmax_cross_entropy_with_logits


    def label_counts(self, label_plane):
        # helper function to compute number of each type of label

        counts = tf.math.bincount(tf.cast(label_plane, tf.int32), minlength=3,maxlength=3, dtype=tf.float32)

        return counts

    @tf.function
    def __call__(self, labels, logits):

        # This function receives the inputs labels and logits and returns a loss.\
        # If there is balancing scheme specified, weights are computed on the fly

        with tf.compat.v1.variable_scope('cross_entropy'):

            loss = None

            # Labels and logits are split by detector plane.  TensorFlow's
            # sparse cross-entropy always expects classes in the last axis, so
            # normalize NCHW/NCDHW logits before computing loss or weights.
            for plane_index in [0, 1, 2]:
                plane_labels = labels[plane_index]
                plane_logits = logits[plane_index]
                if self.channels_dim == 1:
                    rank = plane_logits.shape.rank
                    if rank is None:
                        raise ValueError("logit rank must be statically known")
                    permutation = [0] + list(range(2, rank)) + [1]
                    plane_logits = tf.transpose(plane_logits, permutation)

                plane_loss = self._criterion(
                    labels=plane_labels, logits=plane_logits
                )
                if self.balance_type != "none":
                    if self.balance_type == "focal":

                        # Compute this as focal loss:
                        softmax = tf.nn.softmax(plane_logits, axis=-1)
                        one_hot = tf.one_hot(
                            indices=plane_labels, depth=3, axis=-1
                        )
                        weights = (1-softmax)**2
                        weights *= one_hot
                        weights = tf.reduce_sum(input_tensor=weights, axis=-1)


                    elif self.balance_type == "even":
                        counts = self.label_counts(plane_labels)

                        class_weights = tf.constant(0.3333, dtype=tf.float32)/(counts + tf.constant(1.0, dtype=tf.float32))
                        weights = tf.fill(tf.shape(plane_labels), class_weights[0])
                        for class_index in [1, 2]:
                            local_weights = tf.fill(
                                tf.shape(plane_labels), class_weights[class_index]
                            )
                            weights = tf.where(
                                plane_labels == class_index, local_weights, weights
                            )

                        # weights[ ] = class_weights[1]
                        # weights[labels[i] == 2 ] = class_weights[2]
                        pass

                    elif self.balance_type == "light":
                        weights = tf.ones(tf.shape(plane_labels), dtype=tf.float32)
                        for class_index in [1, 2]:
                            if class_index == 1:
                                local_weights = tf.fill(tf.shape(plane_labels), 1.5)
                            else:
                                local_weights = tf.fill(tf.shape(plane_labels), 10.0)
                            weights = tf.where(
                                plane_labels == class_index, local_weights, weights
                            )


                    weights = tf.stop_gradient(weights)

                    # plane_loss = tf.reduce_mean(input_tensor=plane_loss)
                    total_weight = tf.reduce_sum(weights)


                    plane_loss = tf.reduce_sum(weights*plane_loss)

                    plane_loss /= total_weight
                else:
                    plane_loss = tf.reduce_mean(plane_loss)
                if loss is None:
                    loss = plane_loss
                else:
                    loss += plane_loss

            return loss
