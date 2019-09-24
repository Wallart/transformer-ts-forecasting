from dataset.unsupervised.generic_dataset import GenericDataset


class WalkForwardBuilder:

    def __init__(self, dataset, step_size, val_percent=0.2):
        self._step_size = step_size
        self._dataset = dataset
        self._val_percent = val_percent

    def build(self):
        start_idx = 0
        end_idx = self._dataset.width()

        val_pred_start, val_pred_end = end_idx - self._step_size, end_idx
        train_pred_start, train_pred_end = val_pred_start - self._step_size, val_pred_start

        enc_len = train_pred_start - start_idx

        train_enc_start, train_enc_end = start_idx, start_idx + enc_len

        val_enc_start = train_enc_start + self._step_size
        val_enc_end = val_enc_start + enc_len

        train_enc_array = self._dataset.get_block(train_enc_start, train_enc_end)
        train_pred_array = self._dataset.get_block(train_pred_start, train_pred_end)

        val_enc_array = self._dataset.get_block(val_enc_start, val_enc_end)
        val_pred_array = self._dataset.get_block(val_pred_start, val_pred_end)

        num_elt_to_keep = int(len(self._dataset) * self._val_percent)

        val_enc_array = val_enc_array[:num_elt_to_keep, :, :]
        val_pred_array = val_pred_array[:num_elt_to_keep, :, :]

        return GenericDataset(train_enc_array, train_pred_array), GenericDataset(val_enc_array, val_pred_array)
