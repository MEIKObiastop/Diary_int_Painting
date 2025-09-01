from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pytz
import os
from huggingface_hub import InferenceClient
from PIL import Image
import io
import csv


HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in environment variables")
HF_TOKEN = HF_TOKEN.strip()

# --- Flask アプリと DB の設定 ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:0000@localhost/onediary_local')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'testsecret')

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import inspect

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- Flask-Login 設定 ---
login_manager = LoginManager()
login_manager.init_app(app)

def utc_now():
    return datetime.utcnow()

# --- 単語CSV読み込み ---
basedir = os.path.abspath(os.path.dirname(__file__))
word_dict = {}  # 単語とカテゴリーを格納する辞書
csv_path = os.path.join(basedir, 'data', 'feelings.csv')
with open(csv_path, encoding="utf-8-sig") as f:  # BOM対策で utf-8-sig
    reader = csv.DictReader(f)
    for row in reader:
        if not row["word"]:  # 空行ならスキップ
            continue
        word_dict[row["word"]] = row["category"]


class Post(db.Model):
    __tablename__ = 'shapediary_post' # テーブル名を指定
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    date_created = db.Column(db.DateTime, default=utc_now)
    
    user_id = db.Column(db.Integer, db.ForeignKey('shapediary_user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('posts', lazy=True))
    image_path = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f'<Post {self.id}>'

class User(UserMixin, db.Model):
    __tablename__ = 'shapediary_user' # テーブル名を指定
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(128))


# --- 感情分析API呼び出し関数 ---
# --- 感情分析 ---
def analyze_sentiment(text):
    pos_count = 0
    neg_count = 0
    for word,category in word_dict.items():
        if word in text:
            if category == 'positive':
                pos_count += 1
            else:
                neg_count += 1
    total = pos_count + neg_count
    if total == 0:
        return 0.5  # 中立  （辞書にない単語のみの場合）
    return pos_count / total  # ポジティブ度を返す


def generate_prompt(sentiment_score):
    #sentiment_score: 0～1 のポジティブ度
    #0.7～1.0 -> positive
    #0.4～0.69 -> neutral
    #0.0～0.39 -> negative
    
    if sentiment_score >= 0.7:
        sentiment = "positive"
    elif sentiment_score >= 0.4:
        sentiment = "neutral"
    else:
        sentiment = "negative"

    if sentiment == "positive":
        return "A bright yellow and red or orange and lightgreen gradation heart, white background, minimalistic"
    elif sentiment == "negative":
        return "A darkblue and darkgreen or darkred and grey gradation star, white background, minimalistic"
    else:  # neutral
        return "A grey and white and green or white and bluegreen gradation circle, white background, minimalistic"

def generate_image(prompt):
    try:
        client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"]
        )
        # 画像生成（横長に変更）
        image: Image.Image = client.text_to_image(
            prompt,
            model="ByteDance/SDXL-Lightning",
            width=600,   # 横幅
            height=300   # 縦幅
        )

        # PIL.Image -> バイト列
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return img_bytes.read()

    except Exception as e:
        print("Image generation error:", e)
        return None


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    inspector = inspect(db.engine)
    if "shapediary_user" not in inspector.get_table_names():
        print(">>> Creating tables...")
        db.create_all()
        print(">>> Done creating tables!")


@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    # 全件取得（新しい順）
    all_posts = Post.query.filter_by(user_id=current_user.id).order_by(Post.date_created.desc()).all()

    # --- 投稿表示 ---
    posts = []
    for post in all_posts:
        jst_date = post.date_created.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Tokyo'))
        posts.append({
            "content": post.content,
            "date": jst_date,
            "image_path": post.image_path
        })

    return render_template(
        'index.html',
        posts_all=posts
    )



@app.route('/newimage', methods=['GET', 'POST'])
@login_required
def newimage():
    post_id = request.args.get('post_id')
    content = request.args.get('content')

    if post_id:
        post = Post.query.get(post_id)
        content = post.content
    else:
        content = request.form.get('content')
        post = Post(content=content, user_id=current_user.id)
        db.session.add(post)
        db.session.commit()
        post_id = post.id

    # 感情解析
    sentiment_score = analyze_sentiment(content)
    prompt = generate_prompt(sentiment_score)
    image_bytes = generate_image(prompt)
    if not image_bytes:
        flash("画像生成に失敗しました")
        return redirect(url_for("index"))

    image_path = f"static/{current_user.id}_tmp.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    return render_template(
        'newimage.html',
        image_path=image_path,
        content=content,
        post_id=post_id
    )


@app.route('/confirm_image', methods=['POST'])
@login_required
def confirm_image():
    action = request.form['action']
    
    if action == "redo":
        content = request.form['content']
        # redo は post_id が None の場合もあるので、新規作成にリダイレクト
        return redirect(url_for('newimage', content=content))
    
    # action が confirm の場合のみ post_id を使う
    post_id = int(request.form['post_id'])
    post = Post.query.get_or_404(post_id)

    # 一時ファイルを正式なファイル名に変更
    tmp_path = f"static/{current_user.id}_tmp.png"
    final_path = f"static/{current_user.id}_{post.id}.png"
    os.rename(tmp_path, final_path)

    post.image_path = final_path
    db.session.commit()

    return redirect(url_for('index'))


@app.route('/newuser', methods=['GET', 'POST'])
def newuser():
    error_message = None

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username:
            error_message = "ユーザー名を入力してください"
        elif not password:
            error_message = "パスワードを入力してください"
        elif User.query.filter_by(username=username).first():
            error_message = "すでに登録されているユーザー名です"
        else:
            user = User(
                username=username, 
                password=generate_password_hash(password, method='pbkdf2:sha256')
            )
            db.session.add(user)
            db.session.commit()
            return render_template('login.html')

    return render_template('newuser.html', error=error_message)



@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None  # 初期値

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        # ユーザーが存在しないかパスワードが間違っている場合
        if not user or not check_password_hash(user.password, password):
            error_message = "ユーザー名かパスワードが間違っています"
        else:
            login_user(user)
            return redirect(url_for('index'))

    return render_template('login.html', error=error_message)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')


@app.route('/posts', methods=['POST'])
@login_required
def posts():
    content = request.form['diary_entry']
    action = request.form.get('action')

    if action == "generate":
        # POST フォームで newimage に送信
        return render_template(
            'redirect_to_newimage.html',  # 以下のフォームを使った中間ページ
            content=content
        )
    else:
        # 画像なしで投稿
        new_post = Post(content=content, date_created=utc_now(), user_id=current_user.id)
        db.session.add(new_post)
        db.session.commit()
        return redirect(url_for('index'))


@app.route('/edit')
@login_required
def edit():
    posts_all = Post.query.filter_by(user_id=current_user.id).order_by(Post.date_created.desc()).all()
    return render_template('edit.html', posts_all=posts_all)


@app.route('/delete/<int:post_id>', methods=['GET'])
@login_required
def delete(post_id):
    post = Post.query.get_or_404(post_id)
    if post.user_id != current_user.id:
        return redirect(url_for('edit'))
    
    db.session.delete(post)
    db.session.commit()
    return redirect(url_for('edit'))




# ユーザー削除確認画面
@app.route('/user_delete_confirm')
@login_required
def user_delete_confirm():
    return render_template('user_delete_confirm.html')

# 実際に削除するルート
@app.route('/user_delete', methods=['POST'])
@login_required
def user_delete():
    user = current_user
    
    # このユーザーの全投稿を削除
    Post.query.filter_by(user_id=user.id).delete()
    
    # ユーザー自身を削除
    db.session.delete(user)
    db.session.commit()
    
    flash("アカウントと投稿がすべて削除されました。")

    return redirect(url_for('login'))
