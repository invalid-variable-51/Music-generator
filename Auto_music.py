import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, stream
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint

def read_midi(file):
    print(f"Loading Music File: {file}")
    notes = []
    
    try:
        midi = converter.parse(file)
    except Exception as e:
        print(f"Error parsing {file}: {e}")
        return notes
    
    s2 = instrument.partitionByInstrument(midi)
    for part in s2.parts:
        notes_to_parse = part.recurse()
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def filter_repeated_notes(music_data, max_interval=10):
    filtered_data = []
    time_window = max_interval * 4  # Assuming 4 notes per second
    for notes in music_data:
        temp = []
        last_notes = []
        for i, note in enumerate(notes):
            if note not in last_notes:
                temp.append(note)
            last_notes.append(note)
            if len(last_notes) > time_window:
                last_notes.pop(0)
        filtered_data.append(temp)
    return filtered_data

# Path to the directory containing MIDI files
path = r"Give path of your file"

# Read all the filenames
files = [i for i in os.listdir(path) if i.endswith(".mid") or i.endswith(".midi")]

if not files:
    print("No MIDI files found in the directory.")
else:
    print(f"Found {len(files)} MIDI file(s): {files}")

notes_list = [read_midi(os.path.join(path, i)) for i in files]

# Flatten the list of notes
notes_ = [element for sublist in notes_list for element in sublist]

# Filter notes to avoid repetition within 10 seconds
filtered_music = filter_repeated_notes(notes_list)

# No. of unique notes
unique_notes = list(set(notes_))
print(f"Number of unique notes: {len(unique_notes)}")

# Computing frequency of each note
freq = dict(Counter(notes_))
no = [count for _, count in freq.items()]

plt.figure(figsize=(5, 5))
plt.hist(no, bins=20)
plt.xlabel('Note Frequency')
plt.ylabel('Count')
plt.title('Histogram of Note Frequencies')
plt.show()

# Notes with frequency >= 50
frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
print(f"Number of frequent notes: {len(frequent_notes)}")

new_music = []

for notes in filtered_music:
    temp = [note_ for note_ in notes if note_ in frequent_notes]
    new_music.append(temp)

new_music = [seq for seq in new_music if seq]

# Debugging output
print(f"Filtered music sequences count: {len(new_music)}")

no_of_timesteps = 32
x, y = [], []

for note_ in new_music:
    for i in range(0, len(note_) - no_of_timesteps):
        input_ = note_[i:i + no_of_timesteps]
        output = note_[i + no_of_timesteps]
        x.append(input_)
        y.append(output)

print(f"x length: {len(x)}, y length: {len(y)}")

x = np.array(x)
y = np.array(y)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")

if len(x) == 0 or len(y) == 0:
    raise ValueError("No data available for training. Please check the dataset preparation.")

unique_x = list(set(x.ravel()))
x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

x_seq = np.array([[x_note_to_int[j] for j in i] for i in x])

unique_y = list(set(y))
y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
y_seq = np.array([y_note_to_int[i] for i in y])

x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

print(f"x_tr shape: {x_tr.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_tr shape: {y_tr.shape}")
print(f"y_val shape: {y_val.shape}")

def cnn_model(input_shape, n_vocab):
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab, output_dim=100, input_length=input_shape, trainable=True))
    model.add(Conv1D(64, 3, padding='causal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model

n_vocab = len(unique_x)
cnn = cnn_model(input_shape=32, n_vocab=n_vocab)
cnn.summary()

mc = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

history = cnn.fit(x_tr, y_tr, batch_size=128, epochs=50, validation_data=(x_val, y_val), verbose=1, callbacks=[mc])

model = load_model('best_model.keras')

ind = np.random.randint(0, len(x_val) - 1)
random_music = x_val[ind]

predictions = []
for _ in range(120):
    random_music = random_music.reshape(1, no_of_timesteps)
    prob = model.predict(random_music)[0]
    y_pred = np.argmax(prob, axis=0)
    predictions.append(y_pred)
    random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
    random_music = random_music[1:]

x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
predicted_notes = [x_int_to_note[i] for i in predictions]

def convert_to_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                cn = int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music.mid')

convert_to_midi(predicted_notes)
