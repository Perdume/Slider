package org.example;

import java.util.Objects;

public class Move {
    public int[] tryMove(String axis, int pm, int[] location, int[][] board, int[] enemyloc) {
        if (canMove(location, axis, pm) && isthereplayer(location, axis, pm, enemyloc)) {
            return move(axis, pm, location[0], location[1], board);
        }
        return null;
    }

    private boolean isthereplayer(int[] loc, String xy, int pm, int[] enemyloc){
        int row = loc[0];
        int col = loc[1];
        if (Objects.equals(xy, "x")) {
            return col + pm != enemyloc[1] || row != enemyloc[0];
        } else if (Objects.equals(xy, "y")) {
            return col != enemyloc[1] || row + pm != enemyloc[0];
        }
        return true;
    }


    private boolean isWithinBounds(int value) {
        return value >= 0 && value <= 6;
    }

    public boolean canMove(int[] loc, String xy, int pm) {
        int row = loc[0];
        int col = loc[1];

        if (Objects.equals(xy, "x")) {
            return isWithinBounds(col + pm);
        } else if (Objects.equals(xy, "y")) {
            return isWithinBounds(row + pm);
        }
        return false;
    }

    private int[] move(String xy, int pm, int row, int col, int[][] list) {
        if (Objects.equals(xy, "x")) {
            return movePiece(row, col, row, col + pm, 0, pm, list);
        } else if (Objects.equals(xy, "y")) {
            return movePiece(row, col, row + pm, col, pm, 0, list);
        }
        return null;
    }

    private int[] movePiece(int fromRow, int fromCol, int toRow, int toCol, int rowMove, int colMove, int[][] list) {
        if (list[toRow][toCol] == 0) {
            int hatchMove = 0;
            if (rowMove == 0) {
                while (canMove(new int[]{fromRow, fromCol + hatchMove}, "x", colMove)) {
                    if (list[fromRow][fromCol + hatchMove + colMove] != 1) {
                        hatchMove += colMove;
                    } else {
                        break;
                    }
                }
            } else if (colMove == 0) {
                while (canMove(new int[]{fromRow + hatchMove, fromCol}, "y", rowMove)) {
                    if (list[fromRow + hatchMove + rowMove][fromCol] != 1) {
                        hatchMove += rowMove;
                    } else {
                        break;
                    }
                }
            }

            if (rowMove == 0) {
                return new int[]{fromRow, fromCol + hatchMove, 1};
            }
            if (colMove == 0) {
                return new int[]{fromRow + hatchMove, fromCol, 1};
            }
            return null;
        }
        else {
            return new int[]{toRow, toCol, 0};
        }
    }
}
