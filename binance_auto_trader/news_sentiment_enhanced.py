"""
Enhanced News Sentiment Analysis for Higher Win Rate Trading.
Analyzes: FED, War, Trump, Financial Crisis, Macro Events

Key improvements:
1. Bullish/Bearish keyword detection
2. Event impact scoring
3. News-based trading filters
4. Multi-source sentiment aggregation
"""
import requests
from datetime import datetime, timedelta
import re
import numpy as np
from config import NEWS_API_KEY, NEWS_URL, SYMBOL

# ============================================================
# ENHANCED KEYWORD CATEGORIES WITH BULLISH/BEARISH SIGNALS
# ============================================================

# Keywords that typically cause BULLISH price action
BULLISH_KEYWORDS = {
    # FED / Monetary Policy
    'rate cut': 2.0,
    'dovish': 1.8,
    'pause rate': 1.5,
    'quantitative easing': 1.8,
    'fed pivot': 2.0,
    'lower rates': 1.8,
    'stimulus': 1.5,
    'inject liquidity': 1.6,

    # Trump / Politics (historically bullish for crypto)
    'trump crypto': 2.0,
    'trump bitcoin': 2.0,
    'pro crypto': 1.8,
    'crypto friendly': 1.5,
    'strategic reserve': 2.0,
    'bitcoin reserve': 2.5,
    'trump executive order': 1.5,

    # Market Positive
    'etf approved': 2.5,
    'etf approval': 2.5,
    'institutional adoption': 1.8,
    'blackrock': 1.5,
    'fidelity bitcoin': 1.5,
    'mass adoption': 1.5,
    'bullish': 1.2,
    'rally': 1.0,
    'surge': 1.0,
    'breakout': 1.2,
    'all time high': 1.5,
    'ath': 1.3,

    # Regulation Positive
    'regulation clarity': 1.5,
    'legal tender': 2.0,
    'country adopts': 1.8,
    'sec approves': 2.0,
    'approved': 1.2,

    # War De-escalation
    'ceasefire': 1.5,
    'peace talks': 1.3,
    'war ends': 1.8,
    'conflict resolution': 1.3,
}

# Keywords that typically cause BEARISH price action
BEARISH_KEYWORDS = {
    # FED / Monetary Policy
    'rate hike': -2.0,
    'hawkish': -1.8,
    'raise rates': -1.8,
    'quantitative tightening': -1.8,
    'higher for longer': -1.5,
    'inflation high': -1.3,
    'cpi higher': -1.5,
    'hot inflation': -1.5,

    # Trump / Politics (negative scenarios)
    'crypto ban': -2.5,
    'bitcoin ban': -2.5,
    'regulation crackdown': -2.0,
    'sec lawsuit': -1.8,
    'sec charges': -1.8,

    # Market Negative
    'crash': -1.5,
    'dump': -1.3,
    'sell off': -1.5,
    'bearish': -1.2,
    'plunge': -1.5,
    'liquidation': -1.8,
    'billion liquidated': -2.0,
    'mt gox': -1.5,
    'hack': -1.8,
    'exploit': -1.8,
    'rug pull': -2.0,

    # Crisis
    'recession': -1.5,
    'depression': -1.8,
    'bank collapse': -2.0,
    'bank failure': -2.0,
    'bankruptcy': -1.5,
    'default': -1.5,
    'crisis': -1.3,

    # War Escalation
    'war escalat': -1.8,
    'missile strike': -1.5,
    'nuclear': -2.0,
    'invasion': -1.5,
    'military attack': -1.5,
    'sanctions': -1.3,
    'iran attack': -1.8,
    'israel attack': -1.5,
    'russia attack': -1.5,
    'world war': -2.5,
    'conflict escalat': -1.5,
    'attack': -0.8,

    # Inflation / Economic
    'inflation high': -1.5,
    'higher than expected': -1.3,
    'cpi rise': -1.3,
}

# High-impact event keywords (require special handling)
HIGH_IMPACT_EVENTS = {
    'fomc': 2.5,
    'fed meeting': 2.5,
    'powell speak': 2.0,
    'powell speech': 2.0,
    'cpi report': 2.0,
    'jobs report': 1.8,
    'nfp': 1.8,
    'unemployment': 1.5,
    'gdp': 1.5,
    'trump speech': 1.8,
    'trump announce': 1.8,
    'executive order': 1.5,
    'halving': 2.0,
    'bitcoin halving': 2.5,
}

# News sources reliability weights
SOURCE_WEIGHTS = {
    'reuters': 1.5,
    'bloomberg': 1.5,
    'cnbc': 1.3,
    'coindesk': 1.2,
    'cointelegraph': 1.1,
    'decrypt': 1.1,
    'the block': 1.2,
    'wall street journal': 1.4,
    'financial times': 1.4,
    'default': 1.0
}


class EnhancedNewsSentiment:
    def __init__(self, api_key=NEWS_API_KEY):
        self.api_key = api_key
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 300  # 5 minutes cache

    def fetch_news(self, hours_back=6, max_articles=100):
        """Fetch recent news articles."""
        # Check cache
        now = datetime.utcnow()
        if self.cache_time and (now - self.cache_time).seconds < self.cache_duration:
            return self.cache.get('articles', [])

        from_date = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Multiple queries for better coverage
        queries = [
            "bitcoin OR crypto OR cryptocurrency",
            "federal reserve OR fed rate OR powell",
            "trump crypto OR trump bitcoin",
            "war OR conflict OR military",
            "recession OR inflation OR economy"
        ]

        all_articles = []

        for query in queries:
            try:
                params = {
                    "q": query,
                    "from": from_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 30,
                    "apiKey": self.api_key
                }
                response = requests.get(NEWS_URL, params=params, timeout=10)
                articles = response.json().get("articles", [])
                all_articles.extend(articles)
            except Exception as e:
                print(f"[WARN] News fetch failed for query '{query[:30]}...': {e}")

        # Remove duplicates by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        # Update cache
        self.cache['articles'] = unique_articles[:max_articles]
        self.cache_time = now

        return self.cache['articles']

    def get_source_weight(self, source_name):
        """Get reliability weight for news source."""
        if not source_name:
            return SOURCE_WEIGHTS['default']

        source_lower = source_name.lower()
        for source, weight in SOURCE_WEIGHTS.items():
            if source in source_lower:
                return weight
        return SOURCE_WEIGHTS['default']

    def analyze_text(self, text):
        """
        Analyze text for bullish/bearish sentiment.
        Returns: (score, impact, keywords_found)
        """
        if not text:
            return 0, 0, []

        text_lower = text.lower()
        score = 0
        impact = 0
        keywords_found = []

        # Check bullish keywords
        for keyword, weight in BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                impact += abs(weight)
                keywords_found.append((keyword, 'BULLISH', weight))

        # Check bearish keywords
        for keyword, weight in BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                score += weight  # weight is already negative
                impact += abs(weight)
                keywords_found.append((keyword, 'BEARISH', weight))

        # Check high-impact events
        for keyword, weight in HIGH_IMPACT_EVENTS.items():
            if keyword in text_lower:
                impact += weight
                keywords_found.append((keyword, 'HIGH_IMPACT', weight))

        return score, impact, keywords_found

    def get_sentiment_score(self):
        """
        Get overall market sentiment score.
        Returns: dict with score, impact, signal, and details
        """
        articles = self.fetch_news()

        if not articles:
            return {
                'score': 0,
                'impact': 0,
                'signal': 'NEUTRAL',
                'confidence': 0,
                'num_articles': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'high_impact_events': [],
                'recommendation': 'NO_NEWS'
            }

        total_score = 0
        total_impact = 0
        bullish_count = 0
        bearish_count = 0
        high_impact_events = []
        article_details = []

        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', {}).get('name', '')

            # Combine title and description
            full_text = f"{title} {description}"

            # Analyze
            score, impact, keywords = self.analyze_text(full_text)

            if keywords:
                source_weight = self.get_source_weight(source)
                weighted_score = score * source_weight

                total_score += weighted_score
                total_impact += impact

                if score > 0:
                    bullish_count += 1
                elif score < 0:
                    bearish_count += 1

                # Track high impact events
                for kw, kw_type, weight in keywords:
                    if kw_type == 'HIGH_IMPACT':
                        high_impact_events.append(kw)

                article_details.append({
                    'title': title[:100],
                    'score': weighted_score,
                    'source': source,
                    'keywords': [k[0] for k in keywords]
                })

        # Normalize score
        num_relevant = bullish_count + bearish_count
        if num_relevant > 0:
            normalized_score = total_score / num_relevant
        else:
            normalized_score = 0

        # Determine signal
        if normalized_score > 0.5:
            signal = 'STRONG_BULLISH'
        elif normalized_score > 0.2:
            signal = 'BULLISH'
        elif normalized_score < -0.5:
            signal = 'STRONG_BEARISH'
        elif normalized_score < -0.2:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # Calculate confidence based on agreement
        if num_relevant > 0:
            majority = max(bullish_count, bearish_count) / num_relevant
            confidence = majority * min(1.0, num_relevant / 10)  # More articles = more confidence
        else:
            confidence = 0

        # Trading recommendation
        if total_impact > 5 and high_impact_events:
            recommendation = 'WAIT'  # High impact event - too risky
        elif signal in ['STRONG_BULLISH', 'BULLISH'] and confidence > 0.6:
            recommendation = 'LONG_BIAS'
        elif signal in ['STRONG_BEARISH', 'BEARISH'] and confidence > 0.6:
            recommendation = 'SHORT_BIAS'
        else:
            recommendation = 'NEUTRAL'

        return {
            'score': round(normalized_score, 3),
            'raw_score': round(total_score, 3),
            'impact': round(total_impact, 2),
            'signal': signal,
            'confidence': round(confidence, 2),
            'num_articles': len(articles),
            'num_relevant': num_relevant,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'high_impact_events': list(set(high_impact_events)),
            'recommendation': recommendation,
            'top_articles': sorted(article_details, key=lambda x: abs(x['score']), reverse=True)[:5]
        }

    def should_trade(self):
        """
        Determine if it's safe to trade based on news.
        Returns: (should_trade, reason, bias)
        """
        sentiment = self.get_sentiment_score()

        # Don't trade during high-impact events
        if sentiment['high_impact_events']:
            return False, f"High impact events: {sentiment['high_impact_events']}", 'NONE'

        # Don't trade if news is too bearish
        if sentiment['signal'] == 'STRONG_BEARISH' and sentiment['confidence'] > 0.7:
            return False, "Strong bearish sentiment - too risky", 'NONE'

        # Determine bias
        if sentiment['recommendation'] == 'LONG_BIAS':
            return True, "Bullish sentiment supports trading", 'LONG'
        elif sentiment['recommendation'] == 'SHORT_BIAS':
            return True, "Bearish sentiment supports trading", 'SHORT'
        else:
            return True, "Neutral sentiment - use technical analysis", 'NEUTRAL'

    def get_features_for_ml(self):
        """
        Get news features for ML model input.
        Returns array of features.
        """
        sentiment = self.get_sentiment_score()

        return np.array([
            sentiment['score'],                    # Normalized sentiment score
            sentiment['impact'] / 10,              # Normalized impact
            sentiment['confidence'],               # Confidence level
            sentiment['bullish_count'] / 10,       # Bullish article ratio
            sentiment['bearish_count'] / 10,       # Bearish article ratio
            1 if sentiment['recommendation'] == 'LONG_BIAS' else 0,
            1 if sentiment['recommendation'] == 'SHORT_BIAS' else 0,
            1 if sentiment['high_impact_events'] else 0,  # High impact flag
        ])


def analyze_news_enhanced():
    """
    Drop-in replacement for analyze_news() with enhanced analysis.
    Returns single sentiment score for backward compatibility.
    """
    analyzer = EnhancedNewsSentiment()
    sentiment = analyzer.get_sentiment_score()
    return sentiment['score']


def get_news_trading_filter():
    """
    Get news-based trading filter.
    Returns: (can_trade, bias, reason)
    """
    analyzer = EnhancedNewsSentiment()
    return analyzer.should_trade()


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED NEWS SENTIMENT ANALYSIS")
    print("=" * 60)

    analyzer = EnhancedNewsSentiment()
    sentiment = analyzer.get_sentiment_score()

    print(f"\nSentiment Score: {sentiment['score']}")
    print(f"Signal: {sentiment['signal']}")
    print(f"Confidence: {sentiment['confidence']}")
    print(f"Impact Level: {sentiment['impact']}")
    print(f"Recommendation: {sentiment['recommendation']}")

    print(f"\nArticles Analyzed: {sentiment['num_articles']}")
    print(f"Relevant Articles: {sentiment.get('num_relevant', 0)}")
    print(f"Bullish: {sentiment['bullish_count']} | Bearish: {sentiment['bearish_count']}")

    if sentiment.get('high_impact_events'):
        print(f"\nWARNING - HIGH IMPACT EVENTS: {sentiment['high_impact_events']}")

    if sentiment.get('top_articles'):
        print("\nTop Articles by Impact:")
        for i, article in enumerate(sentiment['top_articles'], 1):
            print(f"  {i}. [{article['score']:+.2f}] {article['title'][:60]}...")
            print(f"     Keywords: {article['keywords']}")
    else:
        print("\nNo relevant articles found. Testing with sample headlines...")

        # Test with sample headlines
        test_headlines = [
            "Fed signals potential rate cut in March, markets rally",
            "Trump announces Bitcoin strategic reserve executive order",
            "Russia launches major missile strike on Ukraine",
            "SEC approves spot Bitcoin ETF applications",
            "Major crypto exchange hacked, $100M stolen",
            "Powell: Inflation remains higher than expected",
            "Israel-Iran conflict escalates with new attacks",
        ]

        print("\nSample headline analysis:")
        for headline in test_headlines:
            score, impact, keywords = analyzer.analyze_text(headline)
            signal = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
            kw_list = [k[0] for k in keywords]
            print(f"  [{score:+.1f}] {signal:8} | {headline[:50]}...")
            if kw_list:
                print(f"           Keywords: {kw_list}")

    # Trading recommendation
    can_trade, reason, bias = analyzer.should_trade()
    print(f"\n{'=' * 60}")
    print(f"TRADING RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"Can Trade: {'YES' if can_trade else 'NO'}")
    print(f"Bias: {bias}")
    print(f"Reason: {reason}")
