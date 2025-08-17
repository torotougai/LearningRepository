# REST API設計書

## 目次
- [1. システム概要](#1-システム概要)
- [2. API一覧](#2-api一覧)
- [3. 共通仕様](#3-共通仕様)
  - [3.1 HTTPステータスコード](#31-httpステータスコード)
  - [3.2 URL設計規則](#32-url設計規則)
  - [3.3 認証・認可](#33-認証認可)
  - [3.4 エラーレスポンス](#34-エラーレスポンス)
- [4. API詳細](#4-api詳細)
  - [4.1 ユーザー管理API](#41-ユーザー管理api)
  - [4.2 商品管理API](#42-商品管理api)
  - [4.3 注文管理API](#43-注文管理api)

---

## 1. システム概要

### システム名
[システム名を記載]

### 概要
[システムの概要と目的を記載]

### アーキテクチャ
- **フロントエンド**: [技術スタック]
- **バックエンド**: [技術スタック]
- **データベース**: [データベース種類]
- **認証方式**: [JWT, OAuth2.0 など]

### バージョン
- **APIバージョン**: v1
- **ドキュメントバージョン**: 1.0.0
- **最終更新日**: [日付]

---

## 2. API一覧

### ユーザー管理
| メソッド | エンドポイント | 説明 | 認証 |
|---------|-------------|------|------|
| POST | [/api/v1/users](#post-apiv1users) | ユーザー作成 | 不要 |
| GET | [/api/v1/users](#get-apiv1users) | ユーザー一覧取得 | 必要 |
| GET | [/api/v1/users/{id}](#get-apiv1usersid) | ユーザー詳細取得 | 必要 |
| PUT | [/api/v1/users/{id}](#put-apiv1usersid) | ユーザー更新 | 必要 |
| DELETE | [/api/v1/users/{id}](#delete-apiv1usersid) | ユーザー削除 | 必要 |

### 商品管理
| メソッド | エンドポイント | 説明 | 認証 |
|---------|-------------|------|------|
| POST | [/api/v1/products](#post-apiv1products) | 商品作成 | 必要 |
| GET | [/api/v1/products](#get-apiv1products) | 商品一覧取得 | 不要 |
| GET | [/api/v1/products/{id}](#get-apiv1productsid) | 商品詳細取得 | 不要 |
| PUT | [/api/v1/products/{id}](#put-apiv1productsid) | 商品更新 | 必要 |
| DELETE | [/api/v1/products/{id}](#delete-apiv1productsid) | 商品削除 | 必要 |

### 注文管理
| メソッド | エンドポイント | 説明 | 認証 |
|---------|-------------|------|------|
| POST | [/api/v1/orders](#post-apiv1orders) | 注文作成 | 必要 |
| GET | [/api/v1/orders](#get-apiv1orders) | 注文一覧取得 | 必要 |
| GET | [/api/v1/orders/{id}](#get-apiv1ordersid) | 注文詳細取得 | 必要 |
| PUT | [/api/v1/orders/{id}](#put-apiv1ordersid) | 注文更新 | 必要 |
| DELETE | [/api/v1/orders/{id}](#delete-apiv1ordersid) | 注文削除 | 必要 |

---

## 3. 共通仕様

### 3.1 HTTPステータスコード

| ステータスコード | 説明 | 使用場面 |
|---------------|------|---------|
| 200 | OK | リクエスト成功 |
| 201 | Created | リソース作成成功 |
| 204 | No Content | リクエスト成功（レスポンスボディなし） |
| 400 | Bad Request | リクエスト形式エラー |
| 401 | Unauthorized | 認証エラー |
| 403 | Forbidden | 認可エラー |
| 404 | Not Found | リソースが見つからない |
| 409 | Conflict | リソース競合エラー |
| 422 | Unprocessable Entity | バリデーションエラー |
| 500 | Internal Server Error | サーバー内部エラー |

### 3.2 URL設計規則

#### 基本ルール
- **ベースURL**: `https://api.example.com/api/v1`
- **リソースは名詞の複数形**: `/users`, `/products`, `/orders`
- **階層構造**: `/users/{user_id}/orders`
- **クエリパラメータ**: フィルタリング、ソート、ページング

#### 例
```
GET /api/v1/users?page=1&limit=20&sort=created_at
GET /api/v1/products?category=electronics&price_min=1000
POST /api/v1/users/{user_id}/orders
```

### 3.3 認証・認可

#### 認証方式
- **Bearer Token (JWT)**: `Authorization: Bearer {token}`

#### リクエストヘッダー例
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

### 3.4 エラーレスポンス

#### 統一エラーフォーマット
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "バリデーションエラーが発生しました",
    "details": [
      {
        "field": "email",
        "message": "メールアドレスの形式が正しくありません"
      }
    ]
  }
}
```

---

## 4. API詳細

## 4.1 ユーザー管理API

### POST /api/v1/users
**概要**: 新規ユーザーを作成する

**認証**: 不要

**リクエスト**:
```json
{
  "name": "山田太郎",
  "email": "yamada@example.com",
  "password": "password123",
  "phone": "090-1234-5678"
}
```

**レスポンス** (201 Created):
```json
{
  "id": 1,
  "name": "山田太郎",
  "email": "yamada@example.com",
  "phone": "090-1234-5678",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. リクエストデータのバリデーション
2. メールアドレスの重複チェック
3. パスワードのハッシュ化
4. usersテーブルへのINSERT
5. レスポンス返却

---

### GET /api/v1/users
**概要**: ユーザー一覧を取得する

**認証**: 必要

**クエリパラメータ**:
- `page`: ページ番号 (デフォルト: 1)
- `limit`: 1ページあたりの件数 (デフォルト: 20, 最大: 100)
- `search`: 名前・メールアドレスでの検索
- `sort`: ソート項目 (created_at, name, email)
- `order`: ソート順 (asc, desc)

**レスポンス** (200 OK):
```json
{
  "data": [
    {
      "id": 1,
      "name": "山田太郎",
      "email": "yamada@example.com",
      "phone": "090-1234-5678",
      "created_at": "2024-01-01T10:00:00Z"
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 10,
    "total_count": 200,
    "per_page": 20
  }
}
```

**処理詳細**:
1. 認証トークンの検証
2. クエリパラメータの解析
3. usersテーブルからのSELECT（条件・ソート・ページング適用）
4. レスポンス返却

---

### GET /api/v1/users/{id}
**概要**: 指定されたIDのユーザー詳細を取得する

**認証**: 必要

**パスパラメータ**:
- `id`: ユーザーID

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "name": "山田太郎",
  "email": "yamada@example.com",
  "phone": "090-1234-5678",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. 認証トークンの検証
2. ユーザーIDの存在チェック
3. usersテーブルからのSELECT
4. レスポンス返却

---

### PUT /api/v1/users/{id}
**概要**: 指定されたIDのユーザー情報を更新する

**認証**: 必要

**パスパラメータ**:
- `id`: ユーザーID

**リクエスト**:
```json
{
  "name": "山田花子",
  "phone": "090-5678-1234"
}
```

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "name": "山田花子",
  "email": "yamada@example.com",
  "phone": "090-5678-1234",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T11:00:00Z"
}
```

**処理詳細**:
1. 認証トークンの検証
2. ユーザーIDの存在チェック
3. リクエストデータのバリデーション
4. usersテーブルのUPDATE
5. レスポンス返却

---

### DELETE /api/v1/users/{id}
**概要**: 指定されたIDのユーザーを削除する

**認証**: 必要

**パスパラメータ**:
- `id`: ユーザーID

**レスポンス** (204 No Content):
レスポンスボディなし

**処理詳細**:
1. 認証トークンの検証
2. ユーザーIDの存在チェック
3. 関連データの確認（注文履歴など）
4. usersテーブルからのDELETE
5. レスポンス返却

---

## 4.2 商品管理API

### POST /api/v1/products
**概要**: 新規商品を作成する

**認証**: 必要（管理者権限）

**リクエスト**:
```json
{
  "name": "iPhone 15",
  "description": "最新のiPhoneです",
  "price": 128000,
  "category_id": 1,
  "stock_quantity": 50,
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ]
}
```

**レスポンス** (201 Created):
```json
{
  "id": 1,
  "name": "iPhone 15",
  "description": "最新のiPhoneです",
  "price": 128000,
  "category_id": 1,
  "stock_quantity": 50,
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. 認証・認可の確認
2. リクエストデータのバリデーション
3. カテゴリIDの存在チェック
4. productsテーブルへのINSERT
5. 画像情報の保存
6. レスポンス返却

---

### GET /api/v1/products
**概要**: 商品一覧を取得する

**認証**: 不要

**クエリパラメータ**:
- `page`: ページ番号
- `limit`: 1ページあたりの件数
- `category_id`: カテゴリIDでフィルタ
- `price_min`: 最低価格
- `price_max`: 最高価格
- `search`: 商品名での検索
- `sort`: ソート項目 (price, created_at, name)

**レスポンス** (200 OK):
```json
{
  "data": [
    {
      "id": 1,
      "name": "iPhone 15",
      "price": 128000,
      "category_id": 1,
      "stock_quantity": 50,
      "images": ["https://example.com/image1.jpg"],
      "created_at": "2024-01-01T10:00:00Z"
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 5,
    "total_count": 100,
    "per_page": 20
  }
}
```

**処理詳細**:
1. クエリパラメータの解析
2. productsテーブルからのSELECT（条件・ソート・ページング適用）
3. レスポンス返却

---

### GET /api/v1/products/{id}
**概要**: 指定されたIDの商品詳細を取得する

**認証**: 不要

**パスパラメータ**:
- `id`: 商品ID

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "name": "iPhone 15",
  "description": "最新のiPhoneです",
  "price": 128000,
  "category_id": 1,
  "category_name": "スマートフォン",
  "stock_quantity": 50,
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. 商品IDの存在チェック
2. productsテーブルからのSELECT（カテゴリ情報をJOIN）
3. レスポンス返却

---

### PUT /api/v1/products/{id}
**概要**: 指定されたIDの商品情報を更新する

**認証**: 必要（管理者権限）

**パスパラメータ**:
- `id`: 商品ID

**リクエスト**:
```json
{
  "name": "iPhone 15 Pro",
  "price": 150000,
  "stock_quantity": 30
}
```

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "name": "iPhone 15 Pro",
  "description": "最新のiPhoneです",
  "price": 150000,
  "category_id": 1,
  "stock_quantity": 30,
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T11:00:00Z"
}
```

**処理詳細**:
1. 認証・認可の確認
2. 商品IDの存在チェック
3. リクエストデータのバリデーション
4. productsテーブルのUPDATE
5. レスポンス返却

---

### DELETE /api/v1/products/{id}
**概要**: 指定されたIDの商品を削除する

**認証**: 必要（管理者権限）

**パスパラメータ**:
- `id`: 商品ID

**レスポンス** (204 No Content):
レスポンスボディなし

**処理詳細**:
1. 認証・認可の確認
2. 商品IDの存在チェック
3. 関連データの確認（注文履歴など）
4. productsテーブルからのDELETE
5. レスポンス返却

---

## 4.3 注文管理API

### POST /api/v1/orders
**概要**: 新規注文を作成する

**認証**: 必要

**リクエスト**:
```json
{
  "items": [
    {
      "product_id": 1,
      "quantity": 2,
      "price": 128000
    },
    {
      "product_id": 2,
      "quantity": 1,
      "price": 50000
    }
  ],
  "shipping_address": {
    "postal_code": "100-0001",
    "prefecture": "東京都",
    "city": "千代田区",
    "address": "丸の内1-1-1",
    "building": "東京ビル"
  }
}
```

**レスポンス** (201 Created):
```json
{
  "id": 1,
  "user_id": 1,
  "status": "pending",
  "total_amount": 306000,
  "items": [
    {
      "id": 1,
      "product_id": 1,
      "product_name": "iPhone 15",
      "quantity": 2,
      "price": 128000,
      "subtotal": 256000
    },
    {
      "id": 2,
      "product_id": 2,
      "product_name": "AirPods",
      "quantity": 1,
      "price": 50000,
      "subtotal": 50000
    }
  ],
  "shipping_address": {
    "postal_code": "100-0001",
    "prefecture": "東京都",
    "city": "千代田区",
    "address": "丸の内1-1-1",
    "building": "東京ビル"
  },
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. 認証トークンの検証
2. リクエストデータのバリデーション
3. 商品の在庫チェック
4. トランザクション開始
5. ordersテーブルへのINSERT
6. order_itemsテーブルへのINSERT
7. 在庫数の更新
8. トランザクションコミット
9. レスポンス返却

---

### GET /api/v1/orders
**概要**: 注文一覧を取得する

**認証**: 必要

**クエリパラメータ**:
- `page`: ページ番号
- `limit`: 1ページあたりの件数
- `status`: 注文ステータスでフィルタ (pending, confirmed, shipped, delivered, cancelled)
- `date_from`: 注文日の開始日
- `date_to`: 注文日の終了日

**レスポンス** (200 OK):
```json
{
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "status": "pending",
      "total_amount": 306000,
      "item_count": 2,
      "created_at": "2024-01-01T10:00:00Z"
    }
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 3,
    "total_count": 50,
    "per_page": 20
  }
}
```

**処理詳細**:
1. 認証トークンの検証
2. ユーザー権限の確認（一般ユーザーは自分の注文のみ、管理者は全注文）
3. クエリパラメータの解析
4. ordersテーブルからのSELECT
5. レスポンス返却

---

### GET /api/v1/orders/{id}
**概要**: 指定されたIDの注文詳細を取得する

**認証**: 必要

**パスパラメータ**:
- `id`: 注文ID

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "user_id": 1,
  "status": "pending",
  "total_amount": 306000,
  "items": [
    {
      "id": 1,
      "product_id": 1,
      "product_name": "iPhone 15",
      "quantity": 2,
      "price": 128000,
      "subtotal": 256000
    }
  ],
  "shipping_address": {
    "postal_code": "100-0001",
    "prefecture": "東京都",
    "city": "千代田区",
    "address": "丸の内1-1-1",
    "building": "東京ビル"
  },
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

**処理詳細**:
1. 認証トークンの検証
2. 注文IDの存在チェック
3. アクセス権限の確認（自分の注文または管理者）
4. ordersテーブルからのSELECT（order_itemsをJOIN）
5. レスポンス返却

---

### PUT /api/v1/orders/{id}
**概要**: 指定されたIDの注文ステータスを更新する

**認証**: 必要

**パスパラメータ**:
- `id`: 注文ID

**リクエスト**:
```json
{
  "status": "confirmed"
}
```

**レスポンス** (200 OK):
```json
{
  "id": 1,
  "user_id": 1,
  "status": "confirmed",
  "total_amount": 306000,
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T11:00:00Z"
}
```

**処理詳細**:
1. 認証トークンの検証
2. 注文IDの存在チェック
3. ステータス変更の妥当性チェック
4. ordersテーブルのUPDATE
5. レスポンス返却

---

### DELETE /api/v1/orders/{id}
**概要**: 指定されたIDの注文をキャンセルする

**認証**: 必要

**パスパラメータ**:
- `id`: 注文ID

**レスポンス** (204 No Content):
レスポンスボディなし

**処理詳細**:
1. 認証トークンの検証
2. 注文IDの存在チェック
3. キャンセル可能状態の確認
4. トランザクション開始
5. 注文ステータスを「cancelled」に更新
6. 在庫数の復元
7. トランザクションコミット
8. レスポンス返却

---

## 付録

### データベーススキーマ例

#### usersテーブル
```sql
CREATE TABLE users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  phone VARCHAR(20),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### productsテーブル
```sql
CREATE TABLE products (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  price DECIMAL(10, 2) NOT NULL,
  category_id BIGINT,
  stock_quantity INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### ordersテーブル
```sql
CREATE TABLE orders (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  status ENUM('pending', 'confirmed', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
  total_amount DECIMAL(10, 2) NOT NULL,
  shipping_address JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 変更履歴
| バージョン | 日付 | 変更内容 |
|----------|------|---------|
| 1.0.0 | 2024-01-01 | 初版作成 |
