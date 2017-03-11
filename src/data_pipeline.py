import tensorflow as tf


def preprocess_image(image):

    preprocessed = tf.image.resize_area([image], [300, 300])
    preprocessed = tf.squeeze(preprocessed, axis=[0])
    return preprocessed


def image_pipeline(annotation_queue, batch_size=10):

    csv_line = annotation_queue.dequeue()
    annotation = tf.decode_csv(
        csv_line,
        [[""], [], [], [], [], []],
        name="csv_decoder")

    file_content = tf.read_file(annotation[0])
    jpg_decoder = tf.image.decode_jpeg(
        file_content,
        channels=3,
        name="jpg_decoder")

    height = tf.to_float(tf.shape(jpg_decoder)[0])
    width = tf.to_float(tf.shape(jpg_decoder)[1])
    jpg_preprocessed = preprocess_image(jpg_decoder)

    min_after_dequeue = batch_size*3
    capacity = min_after_dequeue + (batch_size*3)
    batch = tf.train.shuffle_batch(
        [
            jpg_preprocessed,
            annotation[1],
            [[annotation[3]/height,
              annotation[2]/width,
              annotation[5]/height,
              annotation[4]/width]]
        ],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        name="image_batch_producer")

    return batch


def annotation_pipeline(csv_file):

    filename_queue = tf.train.string_input_producer(
       [csv_file],
       name="csv_file_queue")

    file_reader = tf.TextLineReader(
       skip_header_lines=1,
       name="csv_reader")
    _, line = file_reader.read(filename_queue)

    csv_line_queue = tf.train.string_input_producer(
        [line],
        shuffle=False,
        name="csv_line_queue"
    )

    return csv_line_queue
