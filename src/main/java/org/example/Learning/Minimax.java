package org.example.Learning;

import org.example.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Minimax {

    List<int[][]> boardlist;

    List<int[]> PlLocList;

    List<int[]> AILocList;

    List<List<String>> Moves;

    Boolean isCheckMax;

    public Minimax(Boolean checkMax, List<int[][]> boards,List<int[]> pllocs, List<int[]> ailocs){
        isCheckMax = checkMax;
        boardlist = boards;
        PlLocList = pllocs;
        AILocList = ailocs;
    }
    public List<Double> calculate(){
        List<Double> getscores = new ArrayList<>();
        for(int i=0; i<boardlist.size();i++){
            getscores.add(eval(i));
        }
        return getscores;
    }
    private Double eval(int index){
        int[][] board = boardlist.get(index);
        int[] plloc = PlLocList.get(index);
        int[] ailoc = AILocList.get(index);
        AIMoves am = new AIMoves();
        am.getAIMoves(util.createboard(board), util.newloc(plloc), util.newloc(ailoc), new ArrayList<>(), null);
        PlayerMoves pm = new PlayerMoves();
        pm.getPlayerMoves(util.createboard(board), util.newloc(plloc), util.newloc(ailoc), new ArrayList<>(), null);
        return getval(ailoc[0], ailoc[1]) * 0.4 + am.result.size() * 0.6 - pm.result.size() * 0.6 - getval(plloc[0], plloc[1]) * 0.4;
    }

    private int getval(int x, int y){
        int[][] list = new int[][]{
                {-5, -3, -1, 0, -1, -3, -5},
                {-3, -1, 0, 5, 0, -1, -3},
                {-1, 0, 5, 10, 5, 0, -1},
                {0, 5, 10, 1000, 10, 5, 0},
                {-1, 0, 5, 10, 5, 0, -1},
                {-3, -1, 0, 5, 0, -1, -3},
                {1, -3, -1, 0, -1, -3, 1}
        };
        return list[y][x];
    }
}
