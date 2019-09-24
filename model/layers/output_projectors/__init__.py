from model.layers.output_projectors.linear_projector import LinearProjector
from model.layers.output_projectors.quantile_projector import QuantileProjector
from model.layers.output_projectors.student_t_projector import StudentTProjector

projectors = {
    'linear': LinearProjector,
    'quantile': QuantileProjector,
    'studentT': StudentTProjector
}
