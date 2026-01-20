# Modeling-of-Multiphysical-Systems-in-OpenModelica
Modeling of Multiphysical Systems in OpenModelica: An Electric Bicycle

**Neural Network Models**

Two Python implementations were developed to model the human torque law:
• A feed-forward Multi-Layer Perceptron (MLP): model mlp.py
• A recurrent Gated Recurrent Unit (GRU) network: model gru.py

The scripts allow for configurable feature selection, sequence length, and training
parameters. The complete preprocessing, training, validation, and export pipeline was
implemented by the author. The predicted torque can be exported and used as an input
to the OpenModelica bicycle model, enabling a direct comparison between simulations
driven by measured torque and by predicted human torque.

**OpenModelica Bicycle Model**

The longitudinal bicycle dynamics were implemented in OpenModelica using standard
mechanical components. The model, longitudinal bicycle.mo, computes the bicycle
velocity from an applied wheel torque, accounting for aerodynamic drag, rolling resis-
tance, road inclination, and inertial effects.

The implementation serves as a validated baseline model and as a foundation for future
extensions toward an electric bicycle, including motor and battery subsystems
