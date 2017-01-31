from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import cPickle
import copy
import sys
import string

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/new_save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--temperature', type=float, default=1.,
                       help='sampling temperature'
                       '(lower is more conservative, default is 1.0, which is neutral)')
    parser.add_argument('--relevance', type=float, default=-0.2,
                       help='amount of "relevance masking/MMI (disabled by default):"'
                       'higher is more pressure, 0.4 is probably as high as it can go without'
                       'noticeably degrading coherence;'
                       'set to <0 to disable relevance masking')
    args = parser.parse_args()
    sample_main(args)

def get_paths(input_path):
    if os.path.isfile(input_path):
        # Passed a model rather than a checkpoint directory
        model_path = input_path
        save_dir = os.path.dirname(model_path)
    elif os.path.exists(input_path):
        # Passed a checkpoint directory
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError('checkpoint not found in {}.'.format(save_dir))
    else:
        raise ValueError('save_dir is not a valid path.')
    return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')

def sample_main(args):
    model_path, config_path, vocab_path = get_paths(args.save_dir)
    # Arguments passed to sample.py direct us to a saved model.
    # Load the separate arguments by which that model was previously trained.
    # That's saved_args. Use those to load the model.
    with open(config_path) as f:
        saved_args = cPickle.load(f)
    # Separately load chars and vocab from the save directory.
    with open(vocab_path) as f:
        chars, vocab = cPickle.load(f)
    # Create the model from the saved arguments, in inference mode.
    print("Creating model...")
    net = Model(saved_args, True)
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config = tf.ConfigProto(
                    device_count = {'GPU': 0}
                        )
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(net.save_variables_list())
        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, model_path)
        chatbot(net, sess, chars, vocab, args.n, args.relevance, args.temperature)

def initial_state(net, sess):
    # Return freshly initialized model states.
    return sess.run(net.cell.zero_state(1, tf.float32))

def forward_text(net, sess, states, vocab, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            if len(states) == 2:
                # Automatically forward the primary net.
                _, states[0] = net.forward_model(sess, states[0], vocab[char])
                # If the token is newline, reset the mask net state; else, forward it.
                if vocab[char] == '\n':
                    states[1] = initial_state(net, sess)
                else:
                    _, states[1] = net.forward_model(sess, states[1], vocab[char])
            else:
                _, states = net.forward_model(sess, states, vocab[char])
    return states

def scale_prediction(prediction, temperature):
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return scaled_prediction

def sanitize_text(vocab, text):
    return ''.join(i for i in text if i in vocab)

def initial_state_with_relevance_masking(net, sess, relevance):
    if relevance <= 0.: return initial_state(net, sess)
    else: return [initial_state(net, sess), initial_state(net, sess)]

def chatbot(net, sess, chars, vocab, max_length, relevance, temperature):
    states = initial_state_with_relevance_masking(net, sess, relevance)
    while True:
        user_input = sanitize_text(vocab, raw_input('\n> '))
        user_command_entered, reset, states, relevance, temperature = process_user_command(
            user_input, states, relevance, temperature)
        if reset: states = initial_state_with_relevance_masking(net, sess, relevance)
        if user_command_entered: continue
        states = forward_text(net, sess, states, vocab, '> ' + user_input + "\n>")
        computer_response_generator = beam_search_generator(sess=sess, net=net,
            initial_state=copy.deepcopy(states), initial_sample=vocab[' '],
            early_term_token=vocab['\n'], forward_model_fn=forward_with_mask,
            forward_args=(relevance, vocab['\n']), temperature=temperature)
        for i, char_token in enumerate(computer_response_generator):
            print(chars[char_token], end='')
            states = forward_text(net, sess, states, vocab, chars[char_token])
            sys.stdout.flush()
            if i >= max_length: break
        states = forward_text(net, sess, states, vocab, '\n> ')

def process_user_command(user_input, states, relevance, temperature):
    user_command_entered = False
    reset = False
    try:
        if user_input.startswith('--temperature '):
            user_command_entered = True
            temperature = max(0.001, float(user_input[len('--temperature '):]))
            print("[Temperature set to {}]".format(temperature))
        elif user_input.startswith('--relevance '):
            user_command_entered = True
            new_relevance = float(user_input[len('--relevance '):])
            if relevance <= 0. and new_relevance > 0.:
                states = [states, copy.deepcopy(states)]
            elif relevance > 0. and new_relevance <= 0.:
                states = states[0]
            relevance = new_relevance
            print("[Relevance disabled]" if relevance < 0. else "[Relevance set to {}]".format(relevance))
        elif user_input.startswith('--reset'):
            user_command_entered = True
            reset = True
            print("[Model state reset]")
    except ValueError:
        print("[Value error with provided argument.]")
    return user_command_entered, reset, states, relevance, temperature

def consensus_length(beam_outputs, early_term_token):
    for l in xrange(len(beam_outputs[0])):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]: return l, False
    return l, False

def forward_with_mask(sess, net, states, input_sample, forward_args):
    if len(states) != 2:
        # No relevance masking.
        prob, states = net.forward_model(sess, states, input_sample)
        return prob / sum(prob), states
    # states should be a 2-length list: [primary net state, mask net state].
    # forward_args should be a 2-length list/tuple: [relevance, mask_reset_token]
    relevance, mask_reset_token = forward_args
    if input_sample == mask_reset_token:
        # Reset the mask probs when reaching mask_reset_token (newline).
        states[1] = initial_state(net, sess)
    primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
    primary_prob /= sum(primary_prob)
    mask_prob, states[1] = net.forward_model(sess, states[1], input_sample)
    mask_prob /= sum(mask_prob)
    combined_prob = np.exp(np.log(primary_prob) - relevance * np.log(mask_prob))
    # Normalize probabilities so they sum to 1.
    return combined_prob / sum(combined_prob), states

def beam_search_generator(sess, net, initial_state, initial_sample,
    early_term_token, forward_model_fn, forward_args, temperature):
    '''Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).

    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        temperature: how conservatively to sample tokens from each distribution
            (1.0 = neutral, lower means more conservative)
        tot_steps: how many tokens to generate before stopping,
            unless already stopped via early_term_token.
    Returns: a generator to yield a sequence of beam-sampled tokens.'''
    state = initial_state # Stores the best activation states
    sample = initial_sample

    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.

        prediction, state = forward_model_fn(
                    sess, net, state, sample, forward_args)
        prediction = scale_prediction(prediction, temperature)

        token = np.random.choice(len(prediction), p=prediction)

        if (token == early_term_token):
            return
        else:
            yield token

        sample = token

if __name__ == '__main__':
    main()
