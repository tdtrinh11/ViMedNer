from transformers import PreTrainedModel, AutoModel, DebertaModel, BertModel
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from torchcrf import CRF
import torch.autograd as autograd
# import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence



class ModelForTokenClassification(PreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLstmCRF(PreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=0.1, batch_first=True,
                              bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pad_token_label_id=-100
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)

            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
            best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
            outputs = (loss,) + outputs + (best_path,)
        else:
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            if attention_mask is not None:
                mask = (attention_mask == 1)  # & (labels!=-100))
            else:
                mask = torch.ones(logits.shape).bool()  # (labels!=-100)
            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape) * pad_token_label_id
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            outputs = outputs + (best_path,)

        return outputs

class BertCRF(PreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pad_token_label_id=-100
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # if labels is not None:
        #     # loss_fct = nn.CrossEntropyLoss()
        #     pad_mask = (labels != pad_token_label_id)

        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #         # active_loss = attention_mask.view(-1) == 1
        #         # active_logits = logits.view(-1, self.num_labels)[active_loss]
        #         # active_labels = labels.view(-1)[active_loss]
        #         loss_mask = ((attention_mask == 1) & pad_mask)
        #     else:
        #         # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #         loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

        #     crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
        #     crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

        #     loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
        #     # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
        #     # when calculating loss
        #     best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
        #     best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
        #     outputs = (loss,) + outputs + (best_path,)
        # else:
        #     # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
        #     # when calculating loss
        #     if attention_mask is not None:
        #         mask = (attention_mask == 1)  # & (labels!=-100))
        #     else:
        #         mask = torch.ones(logits.shape).bool()  # (labels!=-100)
        #     crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
        #     crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
        #     best_path = self.crf(crf_logits, crf_mask)
        #     temp_labels = torch.ones(mask.shape) * pad_token_label_id
        #     best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
        #     outputs = outputs + (best_path,)

        # return outputs
    

    # -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-27 22:48:17
# Modified from original https://github.com/jiesutd/LatticeLSTM/blob/master/model/crf.py
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
                                                                                                                m_size)  # B * M


class CRFCustom(nn.Module):

    def __init__(self, tagset_size, use_gpu=False):  # average_batch=False,
        super(CRF, self).__init__()
        print("Build CRF...")
        # self.average_batch = average_batch
        self.gpu = use_gpu

        self.START_TAG = -2
        self.STOP_TAG = -1

        self.tagset_size = tagset_size

        # # We add 2 here, because of START_TAG and STOP_TAG
        # self.hidden2tag = nn.Linear(params['hidden_dim'], self.tagset_size + 2)
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        init_transitions[:, self.START_TAG] = -10000.0
        init_transitions[self.STOP_TAG, :] = -10000.0
        if torch.cuda.is_available():
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions, requires_grad=True)

    def init_hidden_cell(self, batch_size, layer_hidden_dim):
        return (torch.randn(2, batch_size, layer_hidden_dim // 2),
                torch.randn(2, batch_size, layer_hidden_dim // 2))

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert (tag_size == self.tagset_size + 2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
            # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size,
                                                                         tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.tagset_size + 2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        # mask = (1 - mask.long()).byte()
        mask = ~(mask)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            # print "cur value:", cur_values.size()
            partition, cur_bp = torch.max(cur_values, 1)
            # print "partsize:",partition.size()
            # exit(0)
            # print partition
            # print cur_bp
            # print "one best, ",idx
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # exit(0)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,
                                                                                                    0).contiguous()  ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size,
                                                                                                    tag_size).expand(
            batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if torch.cuda.is_available():
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, self.STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1, 0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if torch.cuda.is_available():
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data  # detach()
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data.t()  # feili  pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        # return path_score, best_path
        return best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if torch.cuda.is_available():
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:, self.STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len,
                                                                                         batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood(self, feats, mask, tags):
        # nonegative log likelihood
        forward_score, scores = self._calculate_PZ(feats, mask)
        # print('Forward', forward_score)
        gold_score = self._score_sentence(scores, mask, tags)
        # print('Gold', gold_score)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        # if self.average_batch:
        #     return (forward_score - gold_score) / batch_size
        # else:
        return forward_score - gold_score

def to_crf_pad(org_array, org_mask, pad_label_id=-100):
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    crf_pad = (crf_array != pad_label_id)
    # the viterbi decoder function in CRF makes use of multiplicative property of 0, then pads wrong numbers out.
    # Need a*0 = 0 for CRF to work.
    crf_array[~crf_pad] = 0
    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    return out_array
