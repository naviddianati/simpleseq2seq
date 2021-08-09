
from simpleseq2seq import utils, mycallbacks, seq2seq_char_embedding
from tensorflow.keras.callbacks import EarlyStopping

"""shell
!curl -O http://www.manythings.org/anki/fra-eng.zip
!unzip fra-eng.zip
"""


def main():
    epochs = 50  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 100000  # Number of samples to train on.
    batch_size = 64  # Batch size for training.
    
    # Path to the data txt file on disk.
    data_path = "/home/navid/data/keras.io/fra.txt"
        
    # Define a tokenizer object to pass to the parser
    # tokenizer = utils.RegexpTokenizer('\w+|\$[\d\.]+|\S+')         
    tokenizer = utils.CharTokenizer()
    
    # Data object
    data = utils.Seq2SeqDataParser(tokenizer)            
    
    # Open the file with input and output strings
    # on each line, separate by "\t"
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    
    # Read each line and parse each input/output pair
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, output_text, _ = line.split("\t")
        data.parse(input_text, output_text)
    data.compile()
    
    # Instantiate model
    m = seq2seq_char_embedding.Seq2SeqCharWithEmbedding(
        data.max_input_length,
        data.max_output_length,
        data.num_input_tokens,
        data.num_output_tokens,
        latent_dim,
        data.input_token_index,
        data.output_token_index
    )
    
    m.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    m.fit(
        [data.encoder_input_data, data.decoder_input_data],
        data.decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(
                patience=5,
                monitor="val_accuracy"
                ),
            mycallbacks.MyModelCheckpoint(
                export=False,
                monitor="val_accuracy",
                mode='max',
                )
            ]
    )
    
    utils.print_sample_predictions(m, data, 100, 125)


if __name__ == "__main__":
    main()
