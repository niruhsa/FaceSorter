from tqdm import tqdm
import argparse
import face_recognition as fr
import numpy as np
import os

class FaceSorter:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.input = self.kwargs['input']
        self.data = self.kwargs['data']
        self.batch_size = self.kwargs['batch_size']
        self.tolerance = self.kwargs['tolerance']

        self.encodings = []

        if self.load_encodings():
            self.filter_faces()

    def load_encodings(self):
        for _, _, files in os.walk(self.input):
            print('[ OK ] Loading source faces')
            if len(files) == 0:
                print('[ ERROR ] No files to load faces from, please check input directory exists & has images')
                return False

            for f in files:
                file = os.path.join(self.input, f)
                face = fr.load_image_file(file)
                face = fr.face_encodings(face)
                if len(face) > 0:
                    self.encodings.append(face[0])
        return True

    def move_on_error(self, source, dest, img):
        os.rename(source, dest)
        print('[ NON-FATAL ERROR ] Non Fatal Error occured on file: "{}". Moving to sorted folder.'.format(img))

    def filter_faces(self):
        print('[ OK ] Filtering faces... this could take a while!')
        _files = []
        for _, _, files in os.walk(self.data):
            _files = files
        
        os.makedirs(os.path.join(self.data, 'sorted'), exist_ok = True)
        batch = []
        total_batches = 0
        for f in tqdm(_files):
            batch.append(f)

            if len(batch) == self.batch_size or files.index(f) == len(files) - 1:
                bfl = []
                for _img in batch:
                    img = os.path.join(self.data, _img)
                    try:
                        image = fr.load_image_file(img)
                        bfl.append(image)
                    except:
                        self.move_on_error(img.strip(), os.path.join(self.data, 'sorted', _img.strip()), _img.strip())
                        del batch[batch.index(_img)]
                
                bfl_1 = fr.batch_face_locations(bfl, batch_size = self.batch_size)
                
                for loc in bfl_1:
                    img = batch[bfl_1.index(loc)]
                    image = os.path.join(self.data, img)
                    image = bfl[bfl_1.index(loc)]

                    try:
                        enc = fr.face_encodings(image, loc)
                        dis = fr.face_distance(self.encodings, enc[0])
                        dis = np.min(dis)

                        if dis < self.tolerance:
                            new_path = os.path.join(self.data, 'sorted', img)
                            print('[ OK ] Moving "{}" to "{}"'.format(img, new_path))
                            os.rename(os.path.join(self.data, img.strip()), new_path)
                    except Exception as e:
                        print(e)
                        self.move_on_error(os.path.join(self.data, img.strip()), os.path.join(self.data, 'sorted', img.strip()), img.strip())

                batch = []
                total_batches += 1

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--input', type=str, default='data/faces', help='Input faces to load encodings from')
    args.add_argument('--data', type=str, default='data/sort', help='Faces to sort based on similarity')
    args.add_argument('--batch_size', type=int, default=32, help='Number of images to process in one batch (Speeds up program, but uses more resources). Default is 32.')
    args.add_argument('--tolerance', type=float, default=0.6, help='Tolerance for how similar a face has to be to the dataset to keep. Default is 0.6. (1.0 is the highest, 0.0 is the lowest)')
    args = args.parse_args()

    FaceSorter(**vars(args))



