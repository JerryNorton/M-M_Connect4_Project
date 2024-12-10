import asyncio
import websockets
import numpy as np
from typing import Optional
import os
from ai_test import Connect4AI, load_ai

class Connect4WebsocketPlayer:
    def __init__(self):
        # Load or create AI instance
        if os.path.exists('connect4_ai.pkl'):
            print("Loading pre-trained AI...")
            self.ai = load_ai()
        else:
            print("Creating new AI instance...")
            self.ai = Connect4AI()
            
        # Initialize game state
        self.board = np.zeros((6, 7), dtype=int)
        self.player_number = None  # Will be 1 if creator, 2 if joiner
        
    def make_move(self, column: int, player: int) -> bool:
        """Updates internal board state with a move"""
        for row in range(5, -1, -1):
            if self.board[row][column] == 0:
                self.board[row][column] = player
                return True
        return False
    
    def calculate_move(self, opponent_move: Optional[str] = None) -> int:
        """Calculates next move using the AI"""
        # Update board with opponent's move if provided
        if opponent_move is not None:
            self.make_move(int(opponent_move), 3 - self.player_number)
            
        # Sync AI's board state with current game state
        self.ai.board = self.board.copy()
        
        # Get AI's move, considering whether we're first or second player
        is_first_player = (self.player_number == 1)
        move = self.ai.get_best_move(player=self.player_number, is_first_player=is_first_player)
        
        # Update our board with AI's move
        self.make_move(move, self.player_number)
        
        return move
    
    def display_board(self):
        """Displays current game state"""
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

async def gameloop(socket, created, player):
    """Main game loop handling websocket communication"""
    active = True
    player.player_number = 1 if created else 2
    
    print(f"You are player {player.player_number} ({'X' if created else 'O'})")

    while active:
        try:
            message = (await socket.recv()).split(':')
            print(f"Received message: {message}")
            
            match message[0]:
                case 'GAMESTART':
                    if created:  # We go first if we created the game
                        col = player.calculate_move(None)
                        print(f"Playing column {col}")
                        player.display_board()
                        await socket.send(f'PLAY:{col}')
                        
                case 'OPPONENT':
                    col = player.calculate_move(message[1])
                    print(f"Opponent played {message[1]}")
                    print(f"Playing column {col}")
                    player.display_board()
                    await socket.send(f'PLAY:{col}')
                    
                case 'WIN' | 'LOSS' | 'DRAW' | 'TERMINATED':
                    print(f"Game ended: {message[0]}")
                    player.display_board()
                    active = False
                    
                case 'ERROR':
                    print(f"Error received: {message[1]}")
                    if "Invalid move" in message[1]:
                        # If move was invalid, try another column
                        valid_moves = player.ai.get_valid_moves()
                        if valid_moves:
                            col = valid_moves[0]
                            await socket.send(f'PLAY:{col}')
                    
                case 'ACK':
                    print("Move acknowledged by server")
                    
                case 'ID':
                    print(f"Game ID received: {message[1]}")
                    
                case _:
                    print(f"Unknown message received: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
            active = False
        except Exception as e:
            print(f"Error in game loop: {e}")
            active = False

async def create_game(server):
    """Creates a new game as player 1"""
    try:
        player = Connect4WebsocketPlayer()
        print(f"Connecting to ws://{server}/create")
        async with websockets.connect(f'ws://{server}/create') as socket:
            # Wait for game ID from server
            message = (await socket.recv()).split(':')
            if message[0] == 'ID':
                print(f"Game created with ID: {message[1]}")
            await gameloop(socket, True, player)
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Failed to connect to server: {e}")
    except Exception as e:
        print(f"Error creating game: {e}")

async def join_game(server, game_id):
    """Joins an existing game as player 2"""
    try:
        player = Connect4WebsocketPlayer()
        print(f"Connecting to ws://{server}/join/{game_id}")
        async with websockets.connect(f'ws://{server}/join/{game_id}') as socket:
            await gameloop(socket, False, player)
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Failed to connect to server: {e}")
    except Exception as e:
        print(f"Error joining game: {e}")

if __name__ == '__main__':
    try:
        # Get server information from user
        server = input('Server IP: ').strip()
        protocol = input('Join game or create game? (j/c): ').strip().lower()
        
        match protocol:
            case 'c':
                print("Creating new game...")
                asyncio.run(create_game(server))
            case 'j':
                game_id = input('Game ID: ').strip()
                print(f"Joining game {game_id}...")
                asyncio.run(join_game(server, game_id))
            case _:
                print('Invalid protocol! Please use "j" to join or "c" to create.')
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nGame session ended")