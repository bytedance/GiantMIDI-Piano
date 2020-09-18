import os
import numpy as np
import collections
from mido import MidiFile


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Get note targets
        3. Get pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
        
        # # Prepare targets
        # frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        # onset_roll = np.zeros((frames_num, self.classes_num))
        # offset_roll = np.zeros((frames_num, self.classes_num))
        # reg_onset_roll = np.ones((frames_num, self.classes_num))
        # reg_offset_roll = np.ones((frames_num, self.classes_num))
        # frame_roll = np.zeros((frames_num, self.classes_num))
        # velocity_roll = np.zeros((frames_num, self.classes_num))
        # mask_roll = np.ones((frames_num, self.classes_num))
        # """mask_roll is used for masking out cross segment notes"""

        # reg_pedal_onset_roll = np.ones(frames_num)
        # reg_pedal_offset_roll = np.ones(frames_num)
        # pedal_frame_roll = np.zeros(frames_num)

        # # ------ 2. Get note targets ------
        # # Process note events to target
        # for note_event in note_events:
        #     """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

        #     piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
        #     """There are 88 keys on a piano"""

        #     if 0 <= piano_note <= self.max_piano_note:
        #         bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
        #         fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

        #         if fin_frame >= 0:
        #             frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

        #             offset_roll[fin_frame, piano_note] = 1
        #             velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

        #             # Vector from the center of a frame to ground truth offset
        #             reg_offset_roll[fin_frame, piano_note] = \
        #                 (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

        #             if bgn_frame >= 0:
        #                 onset_roll[bgn_frame, piano_note] = 1

        #                 # Vector from the center of a frame to ground truth onset
        #                 reg_onset_roll[bgn_frame, piano_note] = \
        #                     (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
        #             # Mask out segment notes
        #             else:
        #                 mask_roll[: fin_frame + 1, piano_note] = 0

        # for k in range(self.classes_num):
        #     """Get regression targets"""
        #     reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
        #     reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # # Process unpaired onsets to target
        # for midi_note in buffer_dict.keys():
        #     piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
        #     if 0 <= piano_note <= self.max_piano_note:
        #         bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
        #         mask_roll[bgn_frame :, piano_note] = 0     

        # # ------ 3. Get pedal targets ------
        # # Process pedal events to target
        # for pedal_event in pedal_events:
        #     bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
        #     fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

        #     if fin_frame >= 0:
        #         pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

        #         reg_pedal_offset_roll[fin_frame] = \
        #             (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

        #         if bgn_frame >= 0:
        #             reg_pedal_onset_roll[bgn_frame] = \
        #                 (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # # Get regresssion padal targets
        # reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        # reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        # target_dict = {
        #     'onset_roll': onset_roll, 'offset_roll': offset_roll,
        #     'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
        #     'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
        #     'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
        #     'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
        #     }

        # return target_dict, note_events, pedal_events
        return note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High resolution piano transcription by 
            regressing onset and offset time stamps, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output
