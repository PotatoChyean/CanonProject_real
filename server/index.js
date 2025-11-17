import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import messageRoutes from "./routes/messageRoutes.js";

dotenv.config();
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());
app.use("/api/messages", messageRoutes);

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
