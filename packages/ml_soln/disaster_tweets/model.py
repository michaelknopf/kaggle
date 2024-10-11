from functools import cached_property, cache

from keras_nlp.api.models import DistilBertPreprocessor, DistilBertClassifier
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.optimizers import Adam

from ml_soln.disaster_tweets import ctx


class Model:

    METRICS = ['accuracy']
    PRESET = "distil_bert_base_en_uncased"

    @property
    def model(self) -> DistilBertClassifier:
        return self.classifier

    @cached_property
    def preprocessor(self):
        return DistilBertPreprocessor.from_preset(
            self.PRESET,
            sequence_length=160,
            name="tweets"
        )

    @cached_property
    def classifier(self):
        return DistilBertClassifier.from_preset(
            self.PRESET,
            preprocessor=self.preprocessor,
            num_classes=2
        )

    @cached_property
    def _optimizer(self):
        return Adam(learning_rate=ctx().hyperparams.learning_rate)

    @cached_property
    def _loss(self):
        return SparseCategoricalCrossentropy(from_logits=True)

    @cache
    def _compile(self):
        self.classifier.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=self.METRICS
        )
