import express from "express";
const router = express.Router();

router.post("/predict", async (req, res) => {
  // 테스트용: 랜덤 pass/fail
  const result = Math.random() > 0.5 ? "pass" : "fail";
  res.json({ result });
});

export default router;
