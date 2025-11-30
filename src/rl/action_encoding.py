import chess


class ActionEncoder:
    """
    AlphaZero-style action encoder: 8x8x73 planes => flat 4672 actions.
    """

    TYPE_QUEEN = 0       # 0-55 (8 dirs * 7 dist)
    TYPE_KNIGHT = 56     # 56-63
    TYPE_UNDERPROMO = 64 # 64-72

    queen_dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]
    knight_jumps = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1),
    ]
    promo_dirs = [(1, 0), (1, -1), (1, 1)]  # forward, capture-left, capture-right from white pov
    promo_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

    def encode(self, move: chess.Move, board: chess.Board) -> int:
        """Encode a move to flat index, using canonical (white) perspective."""
        if board.turn == chess.BLACK:
            from_sq = chess.square_mirror(move.from_square)
            to_sq = chess.square_mirror(move.to_square)
        else:
            from_sq = move.from_square
            to_sq = move.to_square

        from_r, from_c = divmod(from_sq, 8)
        to_r, to_c = divmod(to_sq, 8)
        d_r = to_r - from_r
        d_c = to_c - from_c

        # Underpromotion (non-queen promotion)
        if move.promotion and move.promotion != chess.QUEEN:
            try:
                p_idx = self.promo_pieces.index(move.promotion)
            except ValueError as exc:
                raise ValueError(f"Unsupported promotion piece {move.promotion}") from exc
            if d_c == 0:
                direction = 0
            elif d_c == -1:
                direction = 1
            elif d_c == 1:
                direction = 2
            else:
                raise ValueError(f"Invalid underpromotion move {move.uci()}")
            plane = self.TYPE_UNDERPROMO + direction * 3 + p_idx
        # Knight move
        elif (abs(d_r), abs(d_c)) in ((1, 2), (2, 1)):
            try:
                k_idx = self.knight_jumps.index((d_r, d_c))
            except ValueError as exc:
                raise ValueError(f"Invalid knight delta {d_r,d_c}") from exc
            plane = self.TYPE_KNIGHT + k_idx
        else:
            # Sliding move
            dist = max(abs(d_r), abs(d_c))
            if dist == 0:
                raise ValueError("Null move")
            sign_r = d_r // dist
            sign_c = d_c // dist
            try:
                dir_idx = self.queen_dirs.index((sign_r, sign_c))
            except ValueError as exc:
                raise ValueError(f"Invalid slide delta {d_r,d_c}") from exc
            plane = self.TYPE_QUEEN + dir_idx * 7 + (dist - 1)

        return from_sq * 73 + plane

    def decode(self, flat_idx: int, board: chess.Board) -> chess.Move | None:
        """Decode flat action to a move (legal check left to caller)."""
        from_sq_idx = flat_idx // 73
        plane_idx = flat_idx % 73
        from_r, from_c = divmod(from_sq_idx, 8)

        to_r = to_c = None
        promotion = None

        if 0 <= plane_idx < 56:
            dir_idx = (plane_idx - self.TYPE_QUEEN) // 7
            dist = (plane_idx - self.TYPE_QUEEN) % 7 + 1
            dr, dc = self.queen_dirs[dir_idx]
            to_r = from_r + dr * dist
            to_c = from_c + dc * dist
        elif 56 <= plane_idx < 64:
            k_idx = plane_idx - self.TYPE_KNIGHT
            dr, dc = self.knight_jumps[k_idx]
            to_r = from_r + dr
            to_c = from_c + dc
        elif 64 <= plane_idx < 73:
            local_idx = plane_idx - self.TYPE_UNDERPROMO
            direction_idx = local_idx // 3
            piece_idx = local_idx % 3
            dr, dc = self.promo_dirs[direction_idx]
            to_r = from_r + dr
            to_c = from_c + dc
            promotion = self.promo_pieces[piece_idx]
        else:
            return None

        if to_r is None or to_c is None or not (0 <= to_r < 8 and 0 <= to_c < 8):
            return None

        # Flip back to real board orientation
        if board.turn == chess.BLACK:
            final_from = chess.square_mirror(chess.square(from_c, from_r))
            final_to = chess.square_mirror(chess.square(to_c, to_r))
        else:
            final_from = chess.square(from_c, from_r)
            final_to = chess.square(to_c, to_r)

        # auto queen promo if needed
        if promotion is None:
            p = board.piece_at(final_from)
            if p and p.piece_type == chess.PAWN:
                if (board.turn == chess.WHITE and chess.square_rank(final_to) == 7) or (
                    board.turn == chess.BLACK and chess.square_rank(final_to) == 0
                ):
                    promotion = chess.QUEEN
        return chess.Move(final_from, final_to, promotion=promotion)
