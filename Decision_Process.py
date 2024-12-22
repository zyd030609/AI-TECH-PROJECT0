import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from pytorch_net import PolicyValueNet
import pygame
import sys
import copy
import random
from game import move_action2move_id, Game, Board
from mcts import MCTSPlayer
import time
from config import CONFIG






