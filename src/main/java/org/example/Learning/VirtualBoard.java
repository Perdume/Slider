package org.example.Learning;

import org.example.Move;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class VirtualBoard {

    int[][] board;
    int[] playerLoc;
    int[] AILoc;

    int[][] firstboard;
    int[] firstplayerLoc;
    int[] firstAILoc;

    Move mv = new Move();
    public VirtualBoard(int[][] Board, int[] PlayerLoc, int[] EnemyLoc){
        board = Board;
        playerLoc = PlayerLoc;
        AILoc = EnemyLoc;

        firstboard = Board;
        firstplayerLoc = PlayerLoc;
        firstAILoc = EnemyLoc;
    }

    public int[] AIVirtualMove(String xy, int pm){
        int[] sol = mv.tryMove(xy, pm, AILoc, board, playerLoc);
        if(sol==null){
            return null;
        }
        if(sol[2] == 1){
            board[AILoc[0]][AILoc[1]] = 0;
        }
        AILoc = new int[]{sol[0], sol[1]};
        if(sol[2] == 1){
            board[AILoc[0]][AILoc[1]] = 1;
        }
        return sol;
    }

    public int[] PlayerVirtualMove(String xy, int pm){
        int[] sol = mv.tryMove(xy, pm, playerLoc, board, AILoc);
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

    public List<String> getCanAIMove(){
        List<String> ret = new ArrayList<>();
        if(mv.canMove(AILoc, "y", -1)){
            ret.add("w");
        }
        if(mv.canMove(AILoc, "y", 1)){
            ret.add("s");
        }
        if(mv.canMove(AILoc, "x", -1)){
            ret.add("a");
        }
        if(mv.canMove(AILoc, "x", 1)){
            ret.add("d");
        }
        return ret;
    }

    public void resetBoard(){
        board = firstboard;
        playerLoc = firstplayerLoc;
        AILoc = firstAILoc;
    }
    public List<String> getCanplayeMove(){
        List<String> ret = new ArrayList<>();
        if(mv.canMove(playerLoc, "y", -1)){
            ret.add("w");
        }
        if(mv.canMove(playerLoc, "y", 1)){
            ret.add("s");
        }
        if(mv.canMove(playerLoc, "x", -1)){
            ret.add("a");
        }
        if(mv.canMove(playerLoc, "x", 1)){
            ret.add("d");
        }
        return ret;
    }

    public int[][] getBoard(){
        return board;
    }
    public int[] getAILoc(){
        return AILoc;
    }
    public int[] getPlayerLoc(){
        return playerLoc;
    }

}
