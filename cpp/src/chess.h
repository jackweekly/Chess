#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>

// Placeholder bitboard-based chess board.
// This is intentionally minimal; replace with a full implementation or hook to an existing fast bitboard lib.
struct ChessBoard {
    // Bitboards: one per piece/color (not fully implemented)
    std::array<uint64_t, 12> bitboards{};
    bool white_to_move = true;

    ChessBoard();
    explicit ChessBoard(const std::string& fen);

    std::string fen() const;
    bool is_game_over() const;
    std::string result() const; // "1-0","0-1","1/2-1/2"

    std::vector<std::string> legal_moves() const; // UCI strings
    void push(const std::string& uci);
};
