from pathlib import Path
import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.modelinstance import OrthoPDM

path_to_lfpw = Path('./data/lfpw/trainset')
training_shapes = []
for lg in print_progress(mio.import_landmark_files(path_to_lfpw/'*.pts', verbose=True)):
    training_shapes.append(lg['all'])

shape_model = OrthoPDM(training_shapes, max_n_components=None)
print shape_model

shape_model.n_active_components = 20
shape_model.n_active_components = 0.95
print shape_model
