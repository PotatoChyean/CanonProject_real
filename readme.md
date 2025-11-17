

---

## 📁 프로젝트 구조

```

DEEP_CANON/
├── client/        ← Next.js 프론트엔드
└── server/        ← Express 백엔드 + MongoDB 연결

````

---

## 🟢 1. 백엔드 실행 (Express + MongoDB)

### 1️⃣ 환경 변수 설정
`server/.env` 파일 생성:

```env
MONGO_URI=mongodb://localhost:27017/myapp
PORT=5000
````

### 2️⃣ 패키지 설치

```bash
cd server
npm install
```

### 3️⃣ 서버 실행

```bash
node index.js
```

* 서버가 `http://localhost:5000`에서 실행됩니다
* 예시 API 확인: `http://localhost:5000/api/messages`

---

### 4️⃣ 백엔드 코드 예시 (`server/index.js`)

```js
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());

// MongoDB 연결
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("✅ MongoDB 연결 성공"))
.catch(err => console.error("❌ MongoDB 연결 실패:", err));

// 예시 모델
const Message = mongoose.model("Message", new mongoose.Schema({
  text: String,
}));

// API: 메시지 가져오기
app.get("/api/messages", async (req, res) => {
  const messages = await Message.find();
  res.json(messages);
});

// API: 메시지 추가
app.post("/api/messages", async (req, res) => {
  const newMessage = new Message({ text: req.body.text });
  await newMessage.save();
  res.json(newMessage);
});

app.listen(process.env.PORT || 5000, () => console.log("Server running"));
```

---

## 🔵 2. 프론트엔드 실행 (Next.js)

### 1️⃣ 환경 변수 설정

`client/.env.local` 파일 생성:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 2️⃣ 패키지 설치

```bash
cd client
npm install
```

### 3️⃣ 개발 서버 실행

```bash
npm run dev
```

* 브라우저에서 `http://localhost:3000` 접속
* 프론트엔드에서 백엔드 API 호출 가능

---

### 4️⃣ 프론트엔드 코드 예시 (`client/pages/index.js`)

```js
import { useEffect, useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");

  // 메시지 가져오기
  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/messages`)
      .then(res => res.json())
      .then(data => setMessages(data))
      .catch(err => console.error(err));
  }, []);

  // 메시지 추가
  const addMessage = async () => {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const newMsg = await res.json();
    setMessages([...messages, newMsg]);
    setText("");
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Next.js + Express + MongoDB</h1>
      <input
        type="text"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="메시지 입력"
      />
      <button onClick={addMessage}>추가</button>

      <ul>
        {messages.map(msg => (
          <li key={msg._id}>{msg.text}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

## ⚡ 3. MongoDB 설치 및 실행

1. 공식 사이트 다운로드: [MongoDB Community Server](https://www.mongodb.com/try/download/community)
2. 설치 옵션:

   * **Complete** 설치
   * **Service로 설치** 체크
   * **MongoDB Compass** 선택 가능
3. 설치 완료 후 MongoDB 실행

   * 자동 서비스 실행: Windows 서비스에서 확인
   * 수동 실행: `"C:\Program Files\MongoDB\Server\<버전>\bin\mongod.exe"`
4. Express에서 연결 테스트: `mongoose.connect(process.env.MONGO_URI)`

---

## 🔹 4. 실행 순서 요약

1. **MongoDB 실행** (서비스 또는 mongod)
2. **백엔드 실행**

```bash
cd server
node index.js
```

3. **프론트엔드 실행**

```bash
cd client
npm run dev
```

4. 브라우저 → `http://localhost:3000` 접속, 메시지 테스트

---

## 📌 추가 정보

* `.env.local`에서 API URL만 변경하면 배포 환경 대응 가능
* 새로운 패키지 설치 시 `npm install` 필요
* Next.js 프론트와 Express 백엔드는 포트만 맞춰주면 별도 서버로 동작 가능

```

---

이 Markdown 그대로 **README.md**로 저장하면, VS Code에서 깔끔하게 보이고  
명령어, 코드 블록, 폴더 구조까지 모두 정리돼서 바로 따라할 수 있습니다.  

원하면 제가 **이 README를 그림/아이콘 포함해서 한눈에 보는 도식화 버전**도 만들어서  
초보자용 가이드처럼 예쁘게 만들어드릴 수 있어요.  

그거 만들어드릴까요?
```
