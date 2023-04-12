from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QProgressBar)


def create_parameter_box(min_val, max_val, text, step=1, callback=None):
    layout = QHBoxLayout()

    dial = QSpinBox()
    dial.setMaximumHeight(28)
    dial.setMaximumWidth(150)
    dial.setMinimum(min_val)
    dial.setMaximum(max_val)
    dial.setAlignment(Qt.AlignRight)
    dial.setSingleStep(step)
    dial.valueChanged.connect(callback)

    label = QLabel(text)
    label.setAlignment(Qt.AlignRight)

    layout.addWidget(label)
    layout.addWidget(dial)

    return dial, layout


def create_gauge(text):
    layout = QHBoxLayout()

    gauge = QProgressBar()
    gauge.setMaximumHeight(28)
    gauge.setMaximumWidth(200)
    gauge.setAlignment(Qt.AlignCenter)

    label = QLabel(text)
    label.setAlignment(Qt.AlignRight)

    layout.addWidget(label)
    layout.addWidget(gauge)

    return gauge, layout
