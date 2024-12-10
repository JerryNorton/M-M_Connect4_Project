Connect 4 AI with Reinforcement Learning

An AI implementation that plays Connect 4 using Q-learning and minimax search, with websocket support for online play.

Features
  Reinforcement learning through self-play
  Minimax search with alpha-beta pruning
  Position evaluation with weighted board positions
  Move caching for improved performance
  Websocket integration for online games

Installation:
  pip install numpy websockets

Files:

  RL.py - Core AI implementation with training functionality
  connect4.py - Websocket client for online play

  To run with moderator server follow instructions found here: https://github.com/exoRift/mindsmachines-connect4/blob/master/documentation/introduction.md


Usage:
  Training the AI
  bashCopypython RL.py
  This trains the AI for 50,000 episodes and saves to connect4_ai.pkl
  Playing Online
  bashCopypython connect4.py
  Follow prompts to:
    Create a new game ('c')
    Join existing game ('j' + game ID)

Implementation Details:
  Board scoring weighted toward center control
  Q-learning with exploration decay
  Move rewards incentivize faster wins
  Transposition table for caching positions
  Real-time board display

Author:
  Jerry Norton