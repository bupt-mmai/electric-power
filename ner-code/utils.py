import copy
import json
import logging
import numpy as np
import torch.utils.data as Data


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    tokenizer,
    label_list,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
):
    """
    :param examples: List of ``InputExamples``
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param label_list: List of labels.
    :param max_length: Maximum example length
    :param pad_token: 0
    :param pad_token_segment_id: 0
    :return: [(example.guid, input_ids, attention_mask, token_type_ids, label), ......]
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        if example.label is not None:
            label = label_map[example.label]
        else:
            label = None

        if index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

            if example.label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids, attention_mask, token_type_ids, label)
        )

    return features


def convert_examples_to_features_for_ner(
    examples,
    tokenizer,
    label_list,
    max_seq_length=256,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    special_tokens_count=0,
    pad_token=0,
    sequence_a_segment_id=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):

        tokens = []
        label_ids = []
        for word, label in zip(example.text_a, example.label):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
            #    continue
                print(word, word_tokens)
                exit(0)
            tokens.append(word_tokens[0])
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            label_ids.append(label_map[label])

        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # tokens = [cls_token] + tokens + [sep_token]
        # label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
        # segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask,
                          token_type_ids=segment_ids, label=label_ids)
        )
    return features


class BuildDataSet(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        label = np.array(feature.label)

        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.features)


