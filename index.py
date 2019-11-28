import json

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import pandas as pd
from scipy import spatial
import numpy as np
from flask import Flask, request, jsonify
import psycopg2

app = Flask(__name__)

@app.route('/getRecommendations', methods=['post'])
def getRecommendations():
    port = int(os.environ.get('PORT', 5000))
    if port != 5000:
        con = psycopg2.connect(database="dclp5tqns7js4f", user="edmchzqbslnnkd", password="a4a08b5b48f1e0f7bb62e3a955035e51040e37ec6d01ee9eedafece32cd48795",
                               host="ec2-54-221-214-3.compute-1.amazonaws.com", port="5432")
    else:
        con = psycopg2.connect(database="postgres", user="postgres", password="shubham123", host="127.0.0.1", port="5432")
    content = request.get_json(silent=True)
    user_id = content['userId']
    type = content['type']
    already_present = content['alreadyPresentCities']
    cur = con.cursor()
    cur.execute("SELECT id from user_details")
    rows = cur.fetchall()
    id = []
    print(rows)
    for row in rows:
        id.append(row[0])
    if user_id in id:
        table_name = ''
        if type.lower().strip() == 'vacation':
            table_name = 'user_ratings_vacation'
        else:
            table_name = 'user_ratings_relocation'
        cur.execute('select * from '+table_name+' order by user_id')
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
        df.columns = ['User', 'Places', 'rating']
        df = df.drop_duplicates(['User', 'Places'], keep='last')
        print(df)
        R_df = df.pivot(index='User', columns='Places', values='rating').fillna(0)
        R = R_df.as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        from scipy.sparse.linalg import svds
        U, sigma, Vt = svds(R_demeaned, k=2)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)
        index = user_id - min(id)
        sorted_val = list(preds_df.iloc[index])
        col = list(preds_df.columns)
        f = []
        for i in range(0, len(col)):
            f.append({'val':sorted_val[i], 'city': col[i]})
        f.sort(key=lambda x: x['val'], reverse=True)
        final_list = []
        for i in range(0, len(f)):
            if not f[i]['city'] in already_present:
                final_list.append(f[i]['city'])
            if len(final_list) == 3:
                break
        print(final_list)
        final_res = {}
        if type.lower() == 'vacation':
            df_v = pd.read_csv('Vacation_citites.csv')
            final_res = {}
            df_l = list(df_v['Locations'])
            city1 = df_l.index(final_list[0])
            city2 = df_l.index(final_list[1])
            city3 = df_l.index(final_list[2])
            final_res = {
                'city1': {
                    'location': df_v['Locations'][city1].lower().strip(),
                    'budget': df_v['Budget'][city1].lower().strip(),
                    'weather': df_v['Weather'][city1].lower().strip(),
                    'image_url': df_v['Image Url'][city1].lower().strip(),
                    'historical_places': df_v['Historical places'][city1].lower().strip(),
                    'type_of_terrain': df_v['Type of Terrain'][city1].lower().strip(),
                    'family_friendly': df_v['Family Friendly'][city1].lower().strip(),
                    'party_places': df_v['Party Places'][city1].lower().strip(),
                    'cuisine': df_v['Cuisine'][city1].lower().strip(),
                    'transport': df_v['Local Transport'][city1].lower().strip(),
                    'social_env': df_v['Social Enviroment'][city1].lower().strip(),
                    'season': df_v['Season'][city1].lower().strip(),
                    'accomodation': df_v['Accomodation'][city1].lower().strip(),
                },
                'city2': {
                    'location': df_v['Locations'][city2].lower().strip(),
                    'budget': df_v['Budget'][city2].lower().strip(),
                    'weather': df_v['Weather'][city2].lower().strip(),
                    'image_url': df_v['Image Url'][city2].lower().strip(),
                    'historical_places': df_v['Historical places'][city2].lower().strip(),
                    'type_of_terrain': df_v['Type of Terrain'][city2].lower().strip(),
                    'family_friendly': df_v['Family Friendly'][city2].lower().strip(),
                    'party_places': df_v['Party Places'][city2].lower().strip(),
                    'cuisine': df_v['Cuisine'][city2].lower().strip(),
                    'transport': df_v['Local Transport'][city2].lower().strip(),
                    'social_env': df_v['Social Enviroment'][city2].lower().strip(),
                    'season': df_v['Season'][city2].lower().strip(),
                    'accomodation': df_v['Accomodation'][city2].lower().strip(),
                },
                'city3': {
                    'location': df_v['Locations'][city3].lower().strip(),
                    'budget': df_v['Budget'][city3].lower().strip(),
                    'weather': df_v['Weather'][city3].lower().strip(),
                    'image_url': df_v['Image Url'][city3].lower().strip(),
                    'historical_places': df_v['Historical places'][city3].lower().strip(),
                    'type_of_terrain': df_v['Type of Terrain'][city3].lower().strip(),
                    'family_friendly': df_v['Family Friendly'][city3].lower().strip(),
                    'party_places': df_v['Party Places'][city3].lower().strip(),
                    'cuisine': df_v['Cuisine'][city3].lower().strip(),
                    'transport': df_v['Local Transport'][city3].lower().strip(),
                    'social_env': df_v['Social Enviroment'][city3].lower().strip(),
                    'season': df_v['Season'][city3].lower().strip(),
                    'accomodation': df_v['Accomodation'][city3].lower().strip(),
                }
            }
        else:
            pd_r = pd.read_csv('Relocation Cities.csv')
            final_res = {}
            df_l = list(pd_r['City'])
            city1 = df_l.index(final_list[0])
            city2 = df_l.index(final_list[1])
            city3 = df_l.index(final_list[2])
            final_res = {
                'city1': {
                    'city': pd_r['City'][city1].lower().strip(),
                    'taxes': pd_r['Taxes'][city1],
                    'crime_rate': pd_r['crime rate'][city1],
                    'image_url': pd_r['Image link'][city1],
                    'rent': pd_r['Housing Costs (Rent)'][city1],
                    'traffic': pd_r['Traffic'][city1].lower().strip(),
                    'standard_of_education': pd_r['Standard of Education'][city1].lower().strip(),
                    'population_density': pd_r['Population density'][city1].lower().strip(),
                    'living_expense': pd_r['Living Expenses'][city1].lower().strip(),
                    'dist_cities': pd_r['distance from other cities'][city1].lower().strip(),
                }, 'city2': {
                    'city': pd_r['City'][city2].lower().strip(),
                    'taxes': pd_r['Taxes'][city2],
                    'crime_rate': pd_r['crime rate'][city2],
                    'image_url': pd_r['Image link'][city2],
                    'rent': pd_r['Housing Costs (Rent)'][city2],
                    'traffic': pd_r['Traffic'][city2].lower().strip(),
                    'standard_of_education': pd_r['Standard of Education'][city2].lower().strip(),
                    'population_density': pd_r['Population density'][city2].lower().strip(),
                    'living_expense': pd_r['Living Expenses'][city2].lower().strip(),
                    'dist_cities': pd_r['distance from other cities'][city2].lower().strip(),
                }, 'city3': {
                    'city': pd_r['City'][city3].lower().strip(),
                    'taxes': pd_r['Taxes'][city3],
                    'crime_rate': pd_r['crime rate'][city3],
                    'image_url': pd_r['Image link'][city3],
                    'rent': pd_r['Housing Costs (Rent)'][city3],
                    'traffic': pd_r['Traffic'][city3].lower().strip(),
                    'standard_of_education': pd_r['Standard of Education'][city3].lower().strip(),
                    'population_density': pd_r['Population density'][city3].lower().strip(),
                    'living_expense': pd_r['Living Expenses'][city3].lower().strip(),
                    'dist_cities': pd_r['distance from other cities'][city3].lower().strip(),
                }
            }
            print(final_res)

        #R_df = df.pivot(index=0, columns=1, values=2).fillna(0)
    else:
        print('do random')

    return jsonify(final_res), 200

    con.close()
    return 'hello'
@app.route('/getNearestRelocation', methods=['post'])
def getNearestRelocation():
    locations = []
    content = request.get_json(silent=True)
    traffic = {
        'low': 0,
        'moderate': 1,
        'high': 2
    }
    rent = {
        'under $200': 0,
        '$200 to $700': 1,
        'above $700': 2
    }
    education = {
        'graduate': 0,
        'high school': 1,
        'college': 2
    }
    taxes = {
        'above 8.7%': 2,
        '7.5% to 8.7%': 1,
        'under 7.5%': 0
    }
    crime = {
        'low': 0,
        'medium': 1,
        'high': 2
    }
    pop_density = {
        'high': 2,
        'moderate': 1,
        'low': 0
    }
    living_expense = {
        'under 40k': 0,
        '40k to 100k': 1,
        'more than 100k': 2
    }
    dist_from_cities = {
        'under 10 miles': 0,
        '10 to 50 miles': 1,
        'more than 100 miles': 2
    }
    df = pd.read_csv('Relocation Cities.csv')
    for index, row in df.iterrows():
        t = [
                taxes[row['Taxes'].lower().strip()],
                crime[row['crime rate'].lower().strip()],
                rent[row['Housing Costs (Rent)'].lower().strip()],
                traffic[row['Traffic'].lower().strip()],
                education[row['Standard of Education'].lower().strip()],
                pop_density[row['Population density'].lower().strip()],
                living_expense[row['Living Expenses'].lower().strip()],
                dist_from_cities[row['distance from other cities'].lower().strip()]
            ]
        weather = row['weather'].split(' ')
        local_transport = row['access of local transport'].split()
        for i in range(0, len(weather)):
            weather[i] = weather[i].lower().strip()

        if 'cold' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'humid' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'dry' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'warm' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'hot' in weather:
            t.append(1)
        else:
            t.append(-1)

        for i in range(0, len(local_transport)):
            local_transport[i] = local_transport[i].lower().strip()

        if 'road' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'water' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'metro' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'air' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        locations.append({
            'city': row['City'].lower().strip(),
            'taxes': row['Taxes'],
            'crime_rate': row['crime rate'],
            'image_url': row['Image link'],
            'rent': row['Housing Costs (Rent)'],
            'traffic': row['Traffic'].lower().strip(),
            'standard_of_education': row['Standard of Education'].lower().strip(),
            'population_density': row['Population density'].lower().strip(),
            'living_expense': row['Living Expenses'].lower().strip(),
            'dist_cities': row['distance from other cities'].lower().strip(),
            'vector': t
        })

    vector_cmp = [
        taxes[content['taxes'].lower().strip()],
        crime[content['crime_rate'].lower().strip()],
        rent[content['rent'].lower().strip()],
        traffic[content['traffic'].lower().strip()],
        education[content['standard_of_education'].lower().strip()],
        pop_density[content['population_density'].lower().strip()],
        living_expense[content['living_expenses'].lower().strip()],
        dist_from_cities[content['distance_from_other_cities'].lower().strip()]
    ]
    weather = content['weather']
    local_transport = content['access_of_local_transport']
    for i in range(0, len(weather)):
        weather[i] = weather[i].lower().strip()

    if 'cold' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'humid' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'dry' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'warm' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'hot' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    for i in range(0, len(local_transport)):
        local_transport[i] = local_transport[i].lower().strip()

    if 'road' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'water' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'metro' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'air' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    for i in range(0, len(locations)):
        locations[i]['similarity'] = 1 - spatial.distance.cosine(locations[i]['vector'], vector_cmp)
    print(locations)
    res = []
    locations = sorted(locations, key=lambda i: i['similarity'], reverse=True)
    for i in range(0, 3):
        res.append(locations[i])

    for i in range(0, len(res)):
        del res[i]['vector']
    final_res = {
        'city1': res[0],
        'city2': res[1]
    }
    # print(locations)
    return jsonify(final_res), 200

@app.route('/getNearestVacation', methods=['post'])
def getNearestVacation():
    locations = []
    content = request.get_json(silent=True)
    budget = {
        'high': 2,
        'medium': 1,
        'low': 0
    }
    weather = {
        'cold': 0,
        'warm': 1,
        'humid': 2,
        'tropical': 3
    }
    historical = {
        'yes': 1,
        'no': 0
    }
    terrain = {
        'flat': 0,
        'mountain': 1,
        'river': 2,
        'coastal': 3,
        'island': 4,
        'forest': 5,
        'desert': 6
    }

    family_friendly = {
        'yes': 1,
        'no': 0
    }

    cuisine = {
        'continental': 0,
        'local': 1,
    }

    transport = {
        'public': 0,
        'rental': 1
    }

    social_env = {
        'friendly': 0,
        'acceptable': 1
    }

    party = {
        'yes': 1,
        'no': 0
    }
    season = {
        'peak season': 0,
        'off-season': 1
    }

    accomodation = {
        'hotel': 0,
        'homestay': 1,
        'hostel': 2
    }

    df = pd.read_csv('Vacation_citites.csv')
    for index, row in df.iterrows():
        locations.append({
            'location': row['Locations'].lower().strip(),
            'budget':row['Budget'].lower().strip(),
            'weather': row['Weather'].lower().strip(),
            'image_url': row['Image Url'].lower().strip(),
            'historical_places': row['Historical places'].lower().strip(),
            'type_of_terrain': row['Type of Terrain'].lower().strip(),
            'family_friendly': row['Family Friendly'].lower().strip(),
            'party_places': row['Party Places'].lower().strip(),
            'cuisine': row['Cuisine'].lower().strip(),
            'transport': row['Local Transport'].lower().strip(),
            'social_env': row['Social Enviroment'].lower().strip(),
            'season': row['Season'].lower().strip(),
            'accomodation': row['Accomodation'].lower().strip(),
            'vector':[ budget[row['Budget'].lower().strip()],
            weather[row['Weather'].lower().strip()],
            historical[row['Historical places'].lower().strip()],
            terrain[row['Type of Terrain'].lower().strip()],
             family_friendly[row['Family Friendly'].lower().strip()],
             party[row['Party Places'].lower().strip()],
            cuisine[row['Cuisine'].lower().strip()],
            transport[row['Local Transport'].lower().strip()],
           social_env[row['Social Enviroment'].lower().strip()],
            season[row['Season'].lower().strip()],
            accomodation[row['Accomodation'].lower().strip()]]
        })
    vector_cmp = [
        budget[content['budget'].lower().strip()],
        weather[content['weather'].lower().strip()],
        historical[content['historical'].lower().strip()],
        terrain[content['terrain'].lower().strip()],
        family_friendly[content['family_friendly'].lower().strip()],
        party[content['party'].lower().strip()],
        cuisine[content['cuisine'].lower().strip()],
        transport[content['transport'].lower().strip()],
        social_env[content['social_env'].lower().strip()],
        season[content['season'].lower().strip()],
        accomodation[content['accomodation'].lower().strip()]
    ]

    for i in range(0, len(locations)):
        locations[i]['similarity'] = 1 - spatial.distance.cosine(locations[i]['vector'], vector_cmp)

    locations = sorted(locations, key=lambda i: i['similarity'], reverse=True)
    res = []
    for i in range(0, 3):
        res.append(locations[i])

    for i in range(0, len(res)):
        del res[i]['vector']

    final_res = {
        'city1': res[0],
        'city2': res[1]
    }
    #print(locations)
    return jsonify(final_res) , 200

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
