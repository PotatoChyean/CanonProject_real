import { pool } from "../config/db.js";

// 모든 메시지 가져오기
export const getMessages = async () => {
  const [rows] = await pool.query("SELECT * FROM messages");
  return rows;
};

// 메시지 추가
export const addMessage = async (text) => {
  const [result] = await pool.query(
    "INSERT INTO messages (text) VALUES (?)",
    [text]
  );
  return { id: result.insertId, text };
};
