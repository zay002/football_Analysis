import sys
sys.path.append('../')
from utils import get_center_of_bbox,measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance =69
        
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            distance_left = 