package org.example;

import java.util.*;

public class util {
    public static int[][] createboard(int[][] board){
        int[][] nb = new int[7][7];
        for(int i=0;i<board.length;i++){
            System.arraycopy(board[i], 0, nb[i], 0, board[i].length);
        }
        return nb;
    }
    public static int[] newloc(int[] playerloc){
        int[] cl = new int[playerloc.length];
        System.arraycopy(playerloc, 0, cl, 0, playerloc.length);
        return cl;
    }

    public static List<int[]> copiedloc(int[] loc, int count){
        List<int[]> list = new ArrayList<>();
        for(int i=0; i < count; i++){
            list.add(loc);
        }

        return list;
    }

    public static Boolean isEquallocation(int[] loc1, int[] loc2){
        return loc1[0] == loc2[0] && loc1[1] == loc2[1];
    }
}
