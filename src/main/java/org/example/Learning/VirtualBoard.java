package org.example.Learning;

import org.example.Move;

public class VirtualBoard {

    int[][] board;
    int[] playerLoc;
    int[] enemyLoc;

    int[][] firstboard;
    int[] firstplayerLoc;
    int[] firstenemyLoc;

    Move mv = new Move();
    public VirtualBoard(int[][] Board, int[] PlayerLoc, int[] EnemyLoc){
        board = Board;
        playerLoc = PlayerLoc;
        enemyLoc = EnemyLoc;

        firstboard = Board;
        firstplayerLoc = PlayerLoc;
        firstenemyLoc = EnemyLoc;
    }

    public int[] VirtualMove(int dx, int dy){
        int[] sol = mv.tryMove(dx, dy, playerLoc, board, enemyLoc);
        if(sol==null){
            return null;
        }
        if(sol[2] == 1){
            board[playerLoc[0]][playerLoc[1]] = 0;
        }
        playerLoc = new int[]{sol[0], sol[1]};
        if(sol[2] == 1){
            board[playerLoc[0]][playerLoc[1]] = 1;
        }
        return sol;
    }

    public int[][] getBoard(){
        return board;
    }
    public int[] getenemyLoc(){
        return enemyLoc;
    }
    public int[] getPlayerLoc(){
        return playerLoc;
    }

}
