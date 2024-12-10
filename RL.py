# Imports
import numpy as np
import pickle
import random
from typing import Dict, List, Optional, Tuple
from collections import deque

class Connect4AI:
    def __init__(self):
        """Initialize Connect 4 AI with reinforcement learning parameters"""
        # 6 rows x 7 columns game board
        self.board = np.zeros((6, 7), dtype=int)
        # State-action value table for reinforcement learning
        self.q_table = {}  
        # Experience replay buffer for stable learning
        self.memory = deque(maxlen=10000)
        # Learning rate for Q-learning updates
        self.learning_rate = 0.1
        # Discount factor for future rewards
        self.gamma = 0.95
        # Exploration rate for epsilon-greedy policy
        self.epsilon = 0.1
        # Number of experiences to learn from in each replay
        self.batch_size = 32
        # Cache frequently accessed moves for speed
        self.move_cache = {}
        # Initialize positional weights for board evaluation
        self.initialize_weights()
        # Cache for minimax search results
        self.transposition_table = {}

    def initialize_weights(self):
        """Initialize position weights favoring center control"""
        # Higher values for center columns and middle rows
        self.first_player_weights = np.array([
            [3, 4, 5, 7, 5, 4, 3],  # Top row
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3]   # Bottom row
        ])
        # Use same weights for second player
        self.second_player_weights = self.first_player_weights.copy()

    def get_board_hash(self):
        """Get unique string representation of current board state"""
        return str(self.board.tobytes())

    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Learn from random batch of previous experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample random batch of experiences
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Q-learning update: Q(s,a) += α[r + γ max Q(s',a') - Q(s,a)]
                next_hash = str(next_state.tobytes())
                if next_hash in self.q_table:
                    target = reward + self.gamma * max(self.q_table[next_hash])
            
            state_hash = str(state.tobytes())
            if state_hash not in self.q_table:
                self.q_table[state_hash] = np.zeros(7)
            # Update Q-value towards target
            self.q_table[state_hash][action] += self.learning_rate * (target - self.q_table[state_hash][action])

    def get_valid_moves(self) -> List[int]:
        """Return list of columns that aren't full"""
        return [col for col in range(7) if self.board[0][col] == 0]
    
    def make_move(self, column: int, player: int) -> bool:
        """Place player's piece in column. Returns False if column is full"""
        if column not in self.get_valid_moves():
            return False
        
        # Find lowest empty cell in column
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = player
                return True
        return False
    
    def undo_move(self, column: int):
        """Remove top piece from column"""
        for row in range(6):
            if self.board[row][column] != 0:
                self.board[row][column] = 0
                break

    def check_win(self) -> Optional[int]:
        """Check for win in all directions. Returns winning player number or None"""
        # Check horizontal wins
        for row in range(6):
            for col in range(4):
                window = self.board[row, col:col+4]
                if np.all(window == 1):
                    return 1
                if np.all(window == 2):
                    return 2
        
        # Check vertical wins
        for row in range(3):
            for col in range(7):
                window = self.board[row:row+4, col]
                if np.all(window == 1):
                    return 1
                if np.all(window == 2):
                    return 2
        
        # Check diagonal wins
        for row in range(3):
            for col in range(4):
                # Positive slope diagonal
                window = [self.board[row+i][col+i] for i in range(4)]
                if all(x == 1 for x in window):
                    return 1
                if all(x == 2 for x in window):
                    return 2
                
                # Negative slope diagonal
                window = [self.board[row+3-i][col+i] for i in range(4)]
                if all(x == 1 for x in window):
                    return 1
                if all(x == 2 for x in window):
                    return 2
                    
        return None

    def evaluate_position(self, player: int, is_first_player: bool) -> int:
        """Calculate total score for current position"""
        score = 0
        opponent = 3 - player
        
        # Use appropriate weights based on player order
        weights = self.first_player_weights if is_first_player else self.second_player_weights

        # Check for immediate win/loss
        winner = self.check_win()
        if winner == player:
            return 1000000
        elif winner == opponent:
            return -1000000

        # Evaluate all possible windows
        for row in range(6):
            for col in range(4):
                # Score horizontal windows
                window = list(self.board[row, col:col+4])
                score += self._evaluate_window(window, player, is_first_player)
                
                # Score vertical windows
                if row <= 2:
                    window = list(self.board[row:row+4, col])
                    score += self._evaluate_window(window, player, is_first_player)
                
                # Score diagonal windows
                if row <= 2:
                    # Positive slope
                    window = [self.board[row+i][col+i] for i in range(4)]
                    score += self._evaluate_window(window, player, is_first_player)
                    
                    # Negative slope
                    window = [self.board[row+3-i][col+i] for i in range(4)]
                    score += self._evaluate_window(window, player, is_first_player)

        # Add positional scoring based on weights
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == player:
                    score += weights[row][col]
                elif self.board[row][col] == opponent:
                    score -= weights[row][col]

        return score

    def _evaluate_window(self, window: List[int], player: int, is_first_player: bool) -> int:
        """Score a window of 4 positions based on piece configuration"""
        opponent = 3 - player
        score = 0
        
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        # Score completed windows
        if player_count == 4:
            score += 1000
        # Score potential wins
        elif player_count == 3 and empty_count == 1:
            score += 15
        # Score developing positions
        elif player_count == 2 and empty_count == 2:
            score += 5
        
        # Penalize opponent threats
        if opponent_count == 3 and empty_count == 1:
            score -= 95  # Heavy penalty for opponent win threat
        elif opponent_count == 2 and empty_count == 2:
            score -= 10  # Light penalty for opponent development

        return score

    def minimax(self, depth: int, alpha: int, beta: int, maximizing_player: bool, 
                player: int, is_first_player: bool) -> Tuple[Optional[int], int]:
        """Minimax search with alpha-beta pruning and transposition table"""
        board_hash = self.board.tobytes()
        
        # Check if position was previously evaluated
        if board_hash in self.transposition_table:
            stored_depth, stored_move, stored_score = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_move, stored_score

        # Base cases: leaf node or terminal position
        valid_moves = self.get_valid_moves()
        if depth == 0 or not valid_moves or self.check_win() is not None:
            return None, self.evaluate_position(player, is_first_player)

        if maximizing_player:
            max_eval = float('-inf')
            best_move = valid_moves[0]
            
            # Try each move and recursively evaluate
            for move in valid_moves:
                self.make_move(move, player)
                _, eval = self.minimax(depth - 1, alpha, beta, False, player, is_first_player)
                self.undo_move(move)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
            
            # Store result in transposition table
            self.transposition_table[board_hash] = (depth, best_move, max_eval)
            return best_move, max_eval
        else:
            min_eval = float('inf')
            best_move = valid_moves[0]
            
            for move in valid_moves:
                self.make_move(move, 3 - player)
                _, eval = self.minimax(depth - 1, alpha, beta, True, player, is_first_player)
                self.undo_move(move)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
            
            self.transposition_table[board_hash] = (depth, best_move, min_eval)
            return best_move, min_eval

    def get_best_move(self, player: int, is_first_player: bool = True) -> int:
        """Get best move using cached results, Q-table, or minimax search"""
        board_hash = self.get_board_hash()
        
        # Check move cache first
        if board_hash in self.move_cache:
            return self.move_cache[board_hash]

        # Then check Q-table
        if board_hash in self.q_table:
            valid_moves = self.get_valid_moves()
            q_values = [(self.q_table[board_hash][m], m) for m in valid_moves]
            move = max(q_values, key=lambda x: x[0])[1]
            self.move_cache[board_hash] = move
            return move

        # Fall back to minimax search
        move, _ = self.minimax(4, float('-inf'), float('inf'), True, player, is_first_player)
        self.move_cache[board_hash] = move
        return move

    def train(self, episodes=50000):
        """Train AI through self-play with reinforcement learning"""
        for episode in range(episodes):
            # Reset board for new episode
            self.board = np.zeros((6, 7), dtype=int)
            state = self.board.copy()
            done = False
            moves_made = 0
            
            # Randomly decide who goes first
            is_first_player = random.choice([True, False])
            current_player = 1
            
            while not done:
                valid_moves = self.get_valid_moves()
                if not valid_moves:
                    break

                moves_made += 1
                # Increasing penalty for longer games
                move_penalty = -0.5 * moves_made

                if current_player == 1:
                    # AI's move (epsilon-greedy policy)
                    if random.random() < self.epsilon:
                        action = random.choice(valid_moves)
                    else:
                        action = self.get_best_move(1, is_first_player)
                else:
                    # Simulated opponent (mix of random and strategic)
                    if random.random() < 0.3:
                        action = random.choice(valid_moves)
                    else:
                        action = self.get_best_move(2, not is_first_player)

                self.make_move(action, current_player)
                reward = move_penalty

                winner = self.check_win()
                if winner:
                    # Large reward/penalty for win/loss
                    reward += 1000 if winner == 1 else -1000
                    done = True
                elif not self.get_valid_moves():
                    # Penalty for draws
                    reward = -100
                    done = True

                next_state = self.board.copy()
                
                # Only learn from AI's moves
                if current_player == 1:
                    self.remember(state, action, reward, next_state, done)
                    self.replay(self.batch_size)
                
                state = next_state
                current_player = 3 - current_player

            # Decay exploration rate and save checkpoint
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.995)
                print(f"Episode {episode}/{episodes}, Epsilon: {self.epsilon:.3f}")
                save_ai(self, 'connect4_ai_checkpoint.pkl')

    def display_board(self):
        """Print current board state"""
        print("\nCurrent board:")
        for row in self.board:
            print("|", end=" ")
            for cell in row:
                if cell == 0:
                    print(".", end=" ")
                elif cell == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print("|")
        print("-" * 17)
        print("  0 1 2 3 4 5 6  ")

def save_ai(ai: Connect4AI, filename: str = 'connect4_ai.pkl'):
    """Save AI state to file with error handling"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(ai, f)
        print(f"AI successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving AI: {e}")

def load_ai(filename: str = 'connect4_ai.pkl') -> Connect4AI:
    """Load AI state from file with error handling"""
    try:
        with open(filename, 'rb') as f:
            ai = pickle.load(f)
        if not hasattr(ai, 'first_player_weights') or not hasattr(ai, 'second_player_weights'):
            ai = Connect4AI()
        return ai
    except Exception as e:
        print(f"Error loading AI: {e}. Creating new instance.")
        return Connect4AI()

if __name__ == "__main__":
    # Main training routine
    ai = Connect4AI()
    print("Training AI...")
    ai.train(episodes=50000) # Sets the number of training episodes
    print("Training complete!")
    save_ai(ai, 'connect4_ai.pkl') # Final Save