def finetune_language_model(language_model, model_description, corpus, learning_rates):
    """
    Implementation of language model finetuning by unfreezing the layer step by step.
    :param language_model:
    :param model_description:
    :param list learning_rates: list of learning rates.
    :return:
    """
    K.clear_session()
    model_trainer = ModelTrainer(language_model, model_description, corpus)

    for layer in language_model.layers:
        layer.trainable = False

    for i, layer in enumerate(reversed(language_model.layers)):
        layer.trainable = True

        language_model.compile(loss='mse', optimizer=LRMultiplierSGD(lr=learning_rates[i],
                                                                     momentum=0., decay=0.,
                                                                     nesterov=False))

        language_model = model_trainer.train_language_model(batch_size=64, eval_batch_size=10,
                                                            seq_length=50, epochs=1,
                                                            )
    return language_model